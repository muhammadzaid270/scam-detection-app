# Scam Detection OCR (ported from notebook)

This repository contains a modularized version of the code from `OCR.ipynb`.

Files created:

- `app.py` - simple CLI entrypoint and examples of how to call the pipelines
- `ocr_utils.py` - OCR utility functions adapted from the notebook
- `preprocess.py` - image preprocessing utilities
- `region_detect.py` - text-region detection for chat-like screenshots
- `urdu_support.py` - Arabic/Urdu shaping and digit normalization utilities
- `requirements.txt` - suggested pip packages

Notes:

- The original notebook ran some shell installs (e.g., `apt-get` for tesseract). On Windows you must install Tesseract separately if you plan to use `pytesseract`.

- To install dependencies in a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
```

- If you intend to call these functions from a serverless or API integration (e.g. connecting to Gemini), import the functions you need from `ocr_utils` and call them with a PIL image or image bytes.

Quick example (Python):

```python
from PIL import Image
from scam_detection.ocr_utils import extract_text_with_urdu_support

img = Image.open('sample.png')
res = extract_text_with_urdu_support(img)
print(res['clean_text'])
```

If you want, I can also:
- run quick static checks or unit tests locally, or
- add a small Flask/FastAPI wrapper example for easily connecting to Gemini.

Forwarding to Gemini / Deployment
---------------------------------

I added a small FastAPI server in `server.py` and a Gemini client in `gemini_client.py`.

- Start locally (set your API key in env):

```powershell
$env:GEMINI_API_KEY = 'AIzaSyAozuB1e5MoMeJHT0Cc95nYsllJA6Nfq3w'
python -m pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

- Example request (PowerShell `Invoke-RestMethod`):

```powershell
$payload = @{
	member_id = 2
	payload = @{ raw_text = 'Example text'; extracted_fields = @{ phones = @('+12345') } }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri 'http://127.0.0.1:8000/forward' -Method POST -Body $payload -ContentType 'application/json'
```

- Docker build & run:

```powershell
docker build -t scam-ocr:latest .
docker run -e GEMINI_API_KEY=$env:GEMINI_API_KEY -p 8000:8000 scam-ocr:latest
```

Security note: never hard-code API keys into source. Use environment variables or a secrets manager in production.

