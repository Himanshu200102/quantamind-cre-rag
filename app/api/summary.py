# app/api/summary.py
from __future__ import annotations

import os
import time
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["summary"])

DEFAULT_HF_MODEL = os.getenv(
    "HF_SUMMARY_MODEL",
    "meta-llama/Meta-Llama-3-8B-Instruct",  
)

class SummaryRequest(BaseModel):
    text: str = Field(..., description="Raw text to summarize")
    max_new_tokens: int = Field(200, ge=16, le=1024)
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    model: str | None = Field(
        None,
        description="Optional HF model override (e.g., meta-llama/Meta-Llama-3-8B-Instruct)",
    )

class SummaryResponse(BaseModel):
    model: str
    summary: str

def _hf_post_with_retries(
    url: str,
    headers: dict,
    payload: dict,
    retries: int = 3,
    backoff: float = 1.5,
    timeout: int = 60,
) -> requests.Response:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

            # Retry transient “loading” or rate-limit responses
            if resp.status_code in (429, 503, 524, 529):
                last_err = f"{resp.status_code}: {resp.text[:200]}"
                time.sleep(backoff ** attempt)
                continue

            return resp
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(backoff ** attempt)

    raise HTTPException(status_code=502, detail=f"HF call failed after retries: {last_err}")

@router.post("/summary", response_model=SummaryResponse, summary="Summarize via Hugging Face Inference API")
def summarize(req: SummaryRequest) -> SummaryResponse:
    """
    Summarize input text using HF Inference API (text-generation).
    Defaults to meta-llama/Meta-Llama-3-8B-Instruct. You can override with `model`.
    """
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="HF_API_KEY is not set in the environment.")

    model_id = req.model or DEFAULT_HF_MODEL
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = (
        "You are a concise assistant. Summarize the following text in 4–6 sentences. "
        "Preserve key entities, dates, and amounts. Avoid fluff.\n\n"
        f"TEXT:\n{req.text.strip()}"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "return_full_text": False,
        },
    }

    resp = _hf_post_with_retries(url, headers, payload)

    if resp.status_code == 404:
        raise HTTPException(
            status_code=502,
            detail=(
                f"HF error 404: model '{model_id}' not found or not accessible for this token. "
                "Verify the model id and that your HF account has access (gated models require approval)."
            ),
        )
    if resp.status_code in (401, 403):
        raise HTTPException(
            status_code=502,
            detail=(
                f"HF auth error {resp.status_code}: check HF_API_KEY and permissions for '{model_id}'."
            ),
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HF error {resp.status_code}: {resp.text}")

    data = resp.json()
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        summary = str(data[0]["generated_text"]).strip()
    elif isinstance(data, dict) and "generated_text" in data:
        summary = str(data["generated_text"]).strip()
    else:
        summary = str(data)

    return SummaryResponse(model=model_id, summary=summary)
