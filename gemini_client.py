"""Client for calling Google Generative Language (Gemini) REST endpoint using an API key.

This code uses the public REST endpoint for text generation (text-bison style).
It sends a simple prompt and returns the text output. Provide your API key as an
environment variable or parameter. The user-provided API key can be passed directly
to functions but storing it in env var is recommended.
"""
import os
import requests
from typing import Optional, Dict, Any

# Default model — change to desired model name if needed
DEFAULT_MODEL = "text-bison-001"
# Base endpoint for Generative Language API (v1beta2 style)
BASE_URL = "https://generativelanguage.googleapis.com/v1beta2/models"


def build_prompt_from_payload(payload: Dict[str, Any]) -> str:
    """Create a human-friendly prompt for Gemini from member-2 output payload.

    The payload can be any JSON produced by member-2 (e.g., OCR result). This
    function produces a short instruction + the JSON payload so the model can
    analyze, summarize or extract items.
    """
    # Keep prompt concise and instruct the model explicitly what to do
    prompt = (
        "You are an assistant. Analyze the following OCR/extracted data from a scam screenshot "
        "and provide a concise structured response. Return JSON with keys: 'summary', 'risks', "
        "and 'suggested_actions'. Do not include explanations beyond the JSON.\n\n"
        "INPUT_JSON:\n" + str(payload) + "\n\nRespond with valid JSON."
    )
    return prompt


def call_gemini(api_key: str, prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.2, max_output_tokens: int = 512) -> Dict[str, Any]:
    """Call the Generative Language REST API with API key and return parsed response.

    Args:
        api_key: API key string (recommended to keep secret and set via env)
        prompt: the prompt text to send
        model: model name
        temperature: sampling temperature
        max_output_tokens: maximum output tokens

    Returns:
        dict: parsed JSON response with keys 'content' and raw response
    """
    if not api_key:
        raise ValueError("API key is required")

    url = f"{BASE_URL}/{model}:generateText"
    params = {"key": api_key}

    payload = {
        "prompt": {
            "text": prompt
        },
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }

    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, params=params, json=payload, headers=headers, timeout=30)
    try:
        resp.raise_for_status()
    except Exception as e:
        # surface error with details
        raise RuntimeError(f"Gemini API request failed: {e} - {resp.text}")

    data = resp.json()

    # Response format (example): { 'candidates': [ {'output': '...'} ], ... }
    output_text = None
    if isinstance(data, dict):
        # try known shape
        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
            output_text = data["candidates"][0].get("output") or data["candidates"][0].get("content")
        elif "output" in data:
            output_text = data.get("output")
        else:
            # fallback: stringify
            output_text = str(data)
    else:
        output_text = str(data)

    return {"content": output_text, "raw": data}


def send_member2_output_to_gemini(member2_payload: Dict[str, Any], api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """High-level helper: build a prompt from member2 payload, call Gemini and return parsed reply.

    Args:
        member2_payload: JSON-like dict produced by member 2
        api_key: if not provided, read from env var `GEMINI_API_KEY`
    """
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    prompt = build_prompt_from_payload(member2_payload)
    return call_gemini(api_key=api_key, prompt=prompt, model=model)
