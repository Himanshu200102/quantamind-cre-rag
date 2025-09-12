# app/services/local_llm.py
from __future__ import annotations

import os
import json
import time
import hashlib
import logging
import inspect
from typing import Dict, Tuple

from app.utils.timing import timed

log = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "")
CTX = int(os.getenv("LLM_CONTEXT", "4096"))
N_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))
THREADS = int(os.getenv("LLM_THREADS", "8"))
N_BATCH = int(os.getenv("LLM_N_BATCH", "512"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
TOP_K = int(os.getenv("LLM_TOP_K", "50"))
LLM_BACKEND = os.getenv("LLM_BACKEND", "llama_cpp").lower()   # "trt_llm" or "llama_cpp"

def coerce_json(s: str):
    try:
        return json.loads(s.strip())
    except Exception:
        s = s.strip().strip("`")
        first = s.find("[")
        brace = s.find("{")
        cut = None
        if first != -1 and (brace == -1 or first < brace):
            cut = s[first:]
        elif brace != -1:
            cut = s[brace:]
        if cut:
            try:
                return json.loads(cut)
            except Exception:
                pass
        raise

class LocalLlama:
    """
    Unified gen() wrapper. If LLM_BACKEND=trt_llm, uses TensorRT-LLM; else llama.cpp (GGUF).
    """
    def __init__(self):
        self._resp_cache: Dict[Tuple[int, float, int, str], str] = {}
        self.system_prompt = (
            "You are a careful CRE/lease analysis assistant. "
            "Use only the provided context; never fabricate parties, amounts, or dates."
        )

        self.backend = LLM_BACKEND
        log.info(f"[local_llm] backend={self.backend}")

        if self.backend == "trt_llm":
            # TensorRT-LLM path
            from app.services.trt_runner import TrtLLM
            self.trt = TrtLLM()
            self.llm = None
            self._chat_sig_params = set()
            self._comp_sig_params = set()
        else:
            # llama.cpp path
            if not MODEL_PATH or not os.path.exists(MODEL_PATH):
                raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=CTX,
                n_gpu_layers=N_GPU_LAYERS,
                n_threads=THREADS,
                n_batch=N_BATCH,
                logits_all=False,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            log.info(f"[local_llm] loaded {MODEL_PATH} ctx={CTX} gpu_layers={N_GPU_LAYERS} threads={THREADS} n_batch={N_BATCH}")
            try:
                self._chat_sig_params = set(inspect.signature(self.llm.create_chat_completion).parameters.keys())
            except Exception:
                self._chat_sig_params = set()
            try:
                self._comp_sig_params = set(inspect.signature(self.llm.create_completion).parameters.keys())
            except Exception:
                self._comp_sig_params = set()

    def _safe_chat_completion(self, messages, **proposed):
        # llama.cpp-only
        kwargs = {"messages": messages}
        for k, v in proposed.items():
            if k in self._chat_sig_params:
                kwargs[k] = v
        try:
            return self.llm.create_chat_completion(**kwargs)
        except TypeError as e:
            log.warning(f"[local_llm] chat_completion TypeError -> fallback to completion: {e}")
            sys = ""
            usr = ""
            for m in messages:
                if m.get("role") == "system":
                    sys = m.get("content", "")
                elif m.get("role") == "user":
                    usr = m.get("content", "")
            prompt_text = (sys + "\n\n" + usr).strip()
            comp_kwargs = {"prompt": prompt_text}
            for k in ["temperature", "max_tokens", "top_p", "top_k", "stop"]:
                if k in self._comp_sig_params and k in proposed:
                    comp_kwargs[k] = proposed[k]
            return self.llm.create_completion(**comp_kwargs)

    def gen(self, prompt: str, temp: float = 0.2, max_tokens: int = 512) -> str:
        key = (hash(prompt), temp, max_tokens, self.backend)
        if key in self._resp_cache:
            log.info("[CACHE] llm_prompt hit")
            return self._resp_cache[key]

        if self.backend == "trt_llm":
            # TensorRT-LLM generation
            with timed("llm_answer(total)"):
                with timed("llm_answer(prompt_phase)"):
                    text = self.trt.gen(prompt, temp=temp, max_tokens=max_tokens)
            self._resp_cache[key] = text
            return text

        # llama.cpp path
        with timed("llm_answer(total)"):
            with timed("llm_answer(prompt_phase)"):
                proposed_params = {
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                }
                out = self._safe_chat_completion(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **proposed_params,
                )

        if "choices" in out and out["choices"]:
            msg = out["choices"][0]
            if "message" in msg and "content" in msg["message"]:
                text = msg["message"]["content"]
            else:
                text = msg.get("text", "")
        else:
            text = str(out)

        self._resp_cache[key] = text
        return text
