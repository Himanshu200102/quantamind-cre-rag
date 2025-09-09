# app/services/hf_llm.py
from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any
import requests

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_API_URL = "https://api-inference.huggingface.co/models/{model}"

class HFError(RuntimeError):
    pass

def hf_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    retries: int = 2,
    timeout: int = 60,
) -> str:
    """
    Call Hugging Face Inference API for text generation.
    """
    _model = model or HF_MODEL
    api_key = os.getenv("HUGGINGFACE_API_KEY", HF_API_KEY)
    if not api_key:
        raise HFError("HUGGINGFACE_API_KEY not set")

    url = HF_API_URL.format(model=_model)
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False,
            "do_sample": temperature > 0,
        }
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # API returns a list of generations; each has 'generated_text'
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return str(data[0]["generated_text"]).strip()
                # Some backends return dict with 'generated_text'
                if isinstance(data, dict) and "generated_text" in data:
                    return str(data["generated_text"]).strip()
                raise HFError(f"Unexpected HF response: {data}")
            elif resp.status_code in (503, 529):  # model loading / rate limit
                time.sleep(1.5 + attempt)
                continue
            else:
                raise HFError(f"HF error {resp.status_code}: {resp.text}")
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    raise HFError(f"HF call failed after retries: {last_err}")
