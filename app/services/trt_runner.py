from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

import torch

# TensorRT-LLM v0.21 low-level runner
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

TRT_ENGINE_DIR = os.getenv("TRT_ENGINE_DIR", "/home/himanshu/models/trt/llama-3.2-1b-instruct-fp16")
TRT_TOKENIZER_DIR = os.getenv("TRT_TOKENIZER_DIR", "/home/himanshu/models/hf/llama-3.2-1b-instruct")

# Prevent tokenizers parallel fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class _TRTConfig:
    engine_dir: str = TRT_ENGINE_DIR
    tokenizer_dir: str = TRT_TOKENIZER_DIR
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    top_k: int = int(os.getenv("LLM_TOP_K", "50"))


class TRTGenerator:
    """
    Tiny wrapper around TensorRT-LLM ModelRunner for single-request generation.
    Expects a built engine at TRT_ENGINE_DIR and a matching tokenizer at TRT_TOKENIZER_DIR.
    """

    def __init__(self, cfg: Optional[_TRTConfig] = None):
        self.cfg = cfg or _TRTConfig()
        log.info("[trt-llm] engine=%s tokenizer=%s", self.cfg.engine_dir, self.cfg.tokenizer_dir)

        # Load tokenizer (HF)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_dir, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            # LLaMA often lacks a pad token; set to 0 (TRT-LLM-friendly)
            self.tokenizer.pad_token_id = 0

        # Build low-level runner
        # Note: we do not use mpi/multiproc; single-GPU runner for your VM
        self.runner = ModelRunner.from_dir(self.cfg.engine_dir)
        log.info("[trt-llm] using low-level ModelRunner")

        # Defaults
        self.top_p = self.cfg.top_p
        self.top_k = self.cfg.top_k

    def _format_chat_prompt(self, user_message: str) -> str:
        """
        Format the prompt using Llama 3.2 chat template.
        """
        # Try to use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [{"role": "user", "content": user_message}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                log.warning(f"[trt-llm] Failed to use chat template: {e}")
        
        # Fallback to manual Llama 3.2 format
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def gen(self, prompt: str, temp: float = 0.2, max_tokens: int = 256) -> str:
        """
        Generate with TRT-LLM. Returns decoded text (only the newly generated part).
        """
        try:
            # 1) Encode to a FLAT list of token ids (NOT a 2-D batch tensor)
            #    add_special_tokens=False keeps things simple for instruct models.
            input_ids_list = self.tokenizer.encode(prompt, add_special_tokens=False)
            if not isinstance(input_ids_list, list) or len(input_ids_list) == 0:
                log.warning("[trt-llm] Empty or invalid input_ids_list")
                return ""

            log.debug(f"[trt-llm] input_ids_list length: {len(input_ids_list)}")

            # 2) Convert to 1-D int32 tensor on CUDA
            input_ids = torch.tensor(input_ids_list, dtype=torch.int32, device="cuda")

            # 3) Setup EOS/PAD
            end_id = self.tokenizer.eos_token_id
            if end_id is None:
                end_id = 2  # common for LLaMA
            pad_id = self.tokenizer.pad_token_id or 0

            log.debug(f"[trt-llm] end_id: {end_id}, pad_id: {pad_id}")

            # 4) Call generate with a LIST of 1-D tensors (batch of size 1)
            outputs = self.runner.generate(
                batch_input_ids=[input_ids],       # <-- list of 1-D tensors
                max_new_tokens=int(max_tokens),
                temperature=float(temp),
                top_p=float(self.top_p),
                top_k=int(self.top_k),
                end_id=int(end_id),
                pad_id=int(pad_id),
            )

            log.debug(f"[trt-llm] outputs type: {type(outputs)}")

            # 5) Extract tensor and move to CPU
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                out_ids = outputs[0]
            else:
                out_ids = outputs

            if hasattr(out_ids, "device"):
                out_ids = out_ids.to("cpu")
            
            log.info(f"[trt-llm] out_ids shape: {out_ids.shape if hasattr(out_ids, 'shape') else 'no shape'}, type: {type(out_ids)}")
            
            # Debug: Print raw output
            if hasattr(out_ids, 'tolist'):
                raw_tokens = out_ids.tolist()
                log.info(f"[trt-llm] raw output tokens: {raw_tokens[:20]}..." if len(raw_tokens) > 20 else f"[trt-llm] raw output tokens: {raw_tokens}")
            else:
                log.info(f"[trt-llm] raw output (no tolist): {out_ids}")

            # 6) Handle the 3D tensor shape [batch, beam, sequence] -> [sequence]
            gen_ids = out_ids
            try:
                # Handle 3D tensor: [batch, beam, seq_len] -> [seq_len]
                if hasattr(out_ids, 'ndim'):
                    log.info(f"[trt-llm] tensor dimensions: {out_ids.ndim}, shape: {out_ids.shape}")
                    if out_ids.ndim == 3:
                        # Take first batch, first beam
                        out_ids = out_ids[0, 0]
                        log.info(f"[trt-llm] extracted sequence shape: {out_ids.shape}")
                    elif out_ids.ndim == 2:
                        # Take first batch
                        out_ids = out_ids[0]
                
                # Strip prompt tokens (model returns full sequence including input)
                L_prompt = len(input_ids_list)
                if hasattr(out_ids, 'shape') and out_ids.shape[0] >= L_prompt:
                    gen_ids = out_ids[L_prompt:]
                    log.info(f"[trt-llm] stripped {L_prompt} prompt tokens, remaining: {len(gen_ids)}")
                else:
                    log.info("[trt-llm] using full output (no prompt stripping)")
                    gen_ids = out_ids
                    
            except Exception as e:
                log.warning(f"[trt-llm] Error in tensor processing: {e}")
                gen_ids = out_ids  # fallback: use raw output

            # 7) Convert to token list and decode
            try:
                if hasattr(gen_ids, 'tolist'):
                    token_ids = gen_ids.tolist()
                elif isinstance(gen_ids, (list, tuple)):
                    token_ids = list(gen_ids)
                else:
                    log.error(f"[trt-llm] Unknown gen_ids type: {type(gen_ids)}")
                    return "Error: Unable to process generated tokens"
                
                # Ensure flat list of integers
                if isinstance(token_ids, list) and len(token_ids) > 0:
                    # Handle deeply nested structures
                    while isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple)):
                        token_ids = [item for sublist in token_ids for item in sublist]
                    
                    # Convert to integers
                    token_ids = [int(token_id) for token_id in token_ids if isinstance(token_id, (int, float))]
                
                if not token_ids:
                    log.warning("[trt-llm] No tokens to decode after processing")
                    return ""
                
                log.info(f"[trt-llm] decoding {len(token_ids)} tokens: {token_ids[:10]}..." if len(token_ids) > 10 else f"[trt-llm] decoding {len(token_ids)} tokens: {token_ids}")
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                log.info(f"[trt-llm] decoded text: '{text[:100]}...'" if len(text) > 100 else f"[trt-llm] decoded text: '{text}'")
                return text.strip()
                
            except Exception as e:
                log.error(f"[trt-llm] Error in decoding: {e}")
                return f"Error: Failed to decode tokens - {str(e)}"

        except Exception as e:
            log.error(f"[trt-llm] Generation failed: {e}")
            return f"Error: Generation failed - {str(e)}"


# Singleton helper
_trt_singleton: Optional[TRTGenerator] = None

def get_trt_llm() -> TRTGenerator:
    global _trt_singleton
    if _trt_singleton is None:
        _trt_singleton = TRTGenerator()
    return _trt_singleton