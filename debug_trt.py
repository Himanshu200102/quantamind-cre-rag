#!/usr/bin/env python3
"""
Direct debug script for TensorRT-LLM generation.
Run this to test TRT generation outside of the web framework.
"""

import os
import torch
import logging
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Paths
TRT_ENGINE_DIR = "/home/himanshu/models/trt/llama-3.2-1b-instruct-fp16"
TRT_TOKENIZER_DIR = "/home/himanshu/models/hf/llama-3.2-1b-instruct"

def test_direct_generation():
    """Test TRT generation with minimal setup"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TRT_TOKENIZER_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 128001
    
    log.info(f"Tokenizer EOS: {tokenizer.eos_token_id}, PAD: {tokenizer.pad_token_id}")
    
    # Load runner
    runner = ModelRunner.from_dir(TRT_ENGINE_DIR)
    log.info("ModelRunner loaded successfully")
    
    # Test prompts
    test_prompts = [
        "Hello",  # Simple
        "What is 2+2?",  # Basic question
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # Formatted
    ]
    
    for i, prompt in enumerate(test_prompts):
        log.info(f"\n--- Test {i+1}: '{prompt[:50]}...' ---")
        
        # Tokenize
        input_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
        log.info(f"Input tokens: {len(input_ids_list)} -> {input_ids_list[:10]}...")
        
        input_ids = torch.tensor(input_ids_list, dtype=torch.int32, device="cuda")
        
        # Generate with different parameters
        try:
            outputs = runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=20,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                end_id=tokenizer.eos_token_id or 128001,
                pad_id=tokenizer.pad_token_id or 128001,
            )
            
            log.info(f"Output type: {type(outputs)}")
            
            if isinstance(outputs, (list, tuple)):
                out_tensor = outputs[0]
            else:
                out_tensor = outputs
                
            if hasattr(out_tensor, 'device'):
                out_tensor = out_tensor.to('cpu')
                
            log.info(f"Output shape: {out_tensor.shape if hasattr(out_tensor, 'shape') else 'no shape'}")
            
            if hasattr(out_tensor, 'tolist'):
                tokens = out_tensor.tolist()
                log.info(f"Generated tokens: {tokens}")
                
                # Try different decoding approaches
                if tokens:
                    # Full decode
                    full_text = tokenizer.decode(tokens, skip_special_tokens=True)
                    log.info(f"Full decode: '{full_text}'")
                    
                    # Skip prompt if it's there
                    if len(tokens) > len(input_ids_list):
                        new_tokens = tokens[len(input_ids_list):]
                        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        log.info(f"New tokens only: '{new_text}'")
                else:
                    log.warning("No tokens generated!")
            
        except Exception as e:
            log.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_direct_generation()