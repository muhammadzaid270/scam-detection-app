"""Run OCR on an image and forward the member-2 style payload to the local forwarder or call Gemini directly.

Usage:
  python forward_ocr.py path/to/image.png --mode urdu --target server --url http://127.0.0.1:8000/forward

Options:
  --target server   : POST payload to the forwarder server
  --target direct   : call Gemini directly using gemini_client
"""
import argparse
import json
import os
import sys
from PIL import Image
import requests

from ocr_utils import extract_text_with_urdu_support, extract_text_whatsapp_aware, extract_text_from_image
from gemini_client import send_member2_output_to_gemini


def build_member2_payload(ocr_result: dict) -> dict:
    return ocr_result


def post_to_server(url: str, payload: dict, api_key: str = None):
    body = {"member_id": 2, "payload": payload}
    if api_key:
        body["api_key"] = api_key
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--mode", choices=["urdu", "simple", "whatsapp"], default="urdu")
    parser.add_argument("--target", choices=["server", "direct"], default="server")
    parser.add_argument("--url", default="http://127.0.0.1:8000/forward", help="Forwarder URL when target=server")
    parser.add_argument("--api-key", default=None, help="Gemini API key (optional; falls back to GEMINI_API_KEY env)")
    args = parser.parse_args()

    # load image
    img = Image.open(args.image)

    if args.mode == 'simple':
        ocr_res = extract_text_from_image(img, engine='easyocr')
    elif args.mode == 'whatsapp':
        ocr_res = extract_text_whatsapp_aware(img, preprocess_func=None)
    else:
        ocr_res = extract_text_with_urdu_support(img, preprocess_func=None)

    payload = build_member2_payload(ocr_res)

    if args.target == 'server':
        api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
        print(f"Posting to server {args.url} (api_key set: {bool(api_key)})...")
        resp = post_to_server(args.url, payload, api_key=api_key)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
    else:
        # call Gemini directly
        api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("API key required for direct Gemini call; set --api-key or GEMINI_API_KEY env var", file=sys.stderr)
            sys.exit(2)
        print("Calling Gemini directly...")
        resp = send_member2_output_to_gemini(payload, api_key=api_key)
        print(json.dumps(resp, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
