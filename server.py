"""FastAPI server that accepts member-2 output and forwards to Gemini.

Endpoint:
 - POST /forward  -> accepts JSON {"member_id": 2, "payload": {...}} and returns Gemini's response

Security: pass the Gemini API key via environment variable `GEMINI_API_KEY` or include in JSON `api_key` (not recommended).
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from gemini_client import send_member2_output_to_gemini

app = FastAPI(title="Scam OCR -> Gemini Forwarder")


class ForwardRequest(BaseModel):
    member_id: int
    payload: Dict[str, Any]
    api_key: Optional[str] = None


@app.post("/forward")
def forward(request: ForwardRequest):
    if request.member_id != 2:
        raise HTTPException(status_code=400, detail="This endpoint expects payloads from member 2")

    api_key = request.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not provided. Set GEMINI_API_KEY env var or include `api_key` in request (not recommended).")

    try:
        resp = send_member2_output_to_gemini(request.payload, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"status": "ok", "gemini": resp}


@app.get("/health")
def health():
    return {"status": "ok"}
