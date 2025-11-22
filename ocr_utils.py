import re
import unicodedata
import numpy as np
from PIL import Image

try:
    import easyocr
except Exception:
    easyocr = None

try:
    import pytesseract
except Exception:
    pytesseract = None

from preprocess import preprocess_image
from region_detect import detect_text_regions
from urdu_support import shape_and_bidi, normalize_digits

# Lazy-initialized reader
_OCR_READER = None
DEFAULT_LANGS = ['en']

def get_reader(langs=None, gpu=False):
    global _OCR_READER
    if easyocr is None:
        return None
    if langs is None:
        langs = DEFAULT_LANGS
    try:
        # reuse if matches
        if _OCR_READER is None or set(langs) - set(_OCR_READER.lang_list):
            _OCR_READER = easyocr.Reader(langs, gpu=gpu)
    except Exception:
        _OCR_READER = easyocr.Reader(DEFAULT_LANGS, gpu=gpu)
    return _OCR_READER


def clean_text(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def extract_text_from_image(image_pil, engine='easyocr'):
    """Simple extraction for whole-image OCR.

    Args:
        image_pil: PIL.Image
        engine: 'easyocr' or 'tesseract'
    Returns:
        dict with raw_text, clean_text, lines, extracted_fields
    """
    img = np.array(image_pil.convert("RGB"))[:, :, ::-1]  # PIL→CV2 (BGR)
    pre = preprocess_image(img)

    raw = ""
    lines = []
    if engine == 'easyocr' and easyocr is not None:
        reader = get_reader()
        results = reader.readtext(pre, detail=1)
        raw = " ".join([r[1] for r in results])
        lines = [{"text": r[1], "bbox": r[0], "conf": float(r[2])} for r in results]
    elif engine == 'tesseract' and pytesseract is not None:
        raw = pytesseract.image_to_string(Image.fromarray(pre), config="--psm 6")
        lines = []
    else:
        raw = ""

    cleaned = clean_text(raw)
    phones = re.findall(r"\+?\d[\d\s\-]{5,}", cleaned)
    emails = re.findall(r"[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cleaned)
    urls = re.findall(r"(https?://\S+|www\.\S+)", cleaned)
    amounts = re.findall(r"(?:Rs\.?|INR|\$)\s?\d[\d,]*", cleaned)

    return {
        "raw_text": raw,
        "clean_text": cleaned,
        "lines": lines,
        "extracted_fields": {
            "phones": phones,
            "emails": emails,
            "urls": urls,
            "amounts": amounts
        }
    }


def extract_text_whatsapp_aware(image_input,
                                engine='easyocr',
                                preprocess_func=None,
                                min_confidence=0.3,
                                supported_langs=None):
    """Region-aware OCR optimized for chat screenshots. Returns structured output.

    image_input may be: filepath, PIL.Image, or numpy ndarray (BGR).
    """
    # load to cv2 BGR
    if isinstance(image_input, str):
        pil = Image.open(image_input).convert('RGB')
        img_bgr = np.array(pil)[:, :, ::-1]
    elif isinstance(image_input, Image.Image):
        pil = image_input.convert('RGB')
        img_bgr = np.array(pil)[:, :, ::-1]
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
        if img_bgr.ndim == 2:
            import cv2
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError("Unsupported image_input type")

    proc_bgr = img_bgr
    if preprocess_func:
        try:
            proc = preprocess_func(img_bgr)
            import cv2
            proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR) if proc.ndim == 2 else proc
        except Exception:
            proc_bgr = img_bgr

    reader = get_reader()
    if reader is None:
        raise RuntimeError("easyocr not available, install easyocr to use this function")

    regions = detect_text_regions(proc_bgr)
    if not regions:
        regions = [proc_bgr]

    lines = []
    assembled_text = []
    for reg in regions:
        try:
            res = reader.readtext(reg, detail=1)
            filtered = []
            for bbox, txt, conf in res:
                confv = float(conf) if conf is not None else 0.0
                if confv >= min_confidence:
                    filtered.append({"bbox": bbox, "text": txt, "conf": confv})
            if filtered:
                filtered_sorted = sorted(filtered, key=lambda r: (r['bbox'][0][1], r['bbox'][0][0]))
                lines.extend(filtered_sorted)
                assembled_text.append(" ".join([f['text'] for f in filtered_sorted]))
        except Exception:
            try:
                single = reader.readtext(reg, detail=0)
                if single:
                    lines.append({"bbox": None, "text": " ".join(single), "conf": None})
                    assembled_text.append(" ".join(single))
            except Exception:
                continue

    raw_text = " ".join(assembled_text)
    clean = re.sub(r"\s+", " ", raw_text).strip()

    phones = re.findall(r"\+?\d[\d\s\-]{5,}", clean)
    emails = re.findall(r"[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", clean)
    urls = re.findall(r"(https?://\S+|www\.\S+)", clean)
    amounts = re.findall(r"(?:Rs\.?|INR|\$)\s?\d[\d,]*", clean)

    return {
        "detected_language": None,
        "raw_text": raw_text,
        "clean_text": clean,
        "lines": lines,
        "extracted_fields": {"phones": phones, "emails": emails, "urls": urls, "amounts": amounts}
    }


def extract_text_with_urdu_support(image_input, min_confidence=0.3, preprocess_func=None):
    """OCR pipeline with auto-language detection and Urdu shaping/bidi support.

    Note: This function assumes `langdetect`, `arabic_reshaper`, and `python-bidi` are available when needed.
    """
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
    except Exception:
        detect = None

    # load image to BGR
    if isinstance(image_input, str):
        pil = Image.open(image_input).convert('RGB')
        img_bgr = np.array(pil)[:, :, ::-1]
    elif isinstance(image_input, Image.Image):
        pil = image_input.convert('RGB')
        img_bgr = np.array(pil)[:, :, ::-1]
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise ValueError("Unsupported image_input type")

    proc_bgr = img_bgr
    if preprocess_func:
        try:
            proc = preprocess_func(img_bgr)
            import cv2
            proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR) if proc.ndim == 2 else proc
        except Exception:
            proc_bgr = img_bgr

    reader = get_reader()
    if reader is None:
        raise RuntimeError("easyocr not available, install easyocr to use this function")

    try:
        quick_res = reader.readtext(proc_bgr, detail=0)
        quick_raw = " ".join(quick_res)
        quick_clean = re.sub(r"\s+", " ", quick_raw).strip()
    except Exception:
        quick_clean = ""

    detected_lang = None
    if detect is not None and quick_clean:
        try:
            detected_lang = detect(quick_clean)
        except Exception:
            detected_lang = 'en'
    else:
        detected_lang = 'en'

    # recreate reader if necessary to include detected language (best-effort)
    try:
        # easyocr language naming is dataset-specific; we simply ensure reader exists
        reader = get_reader()
    except Exception:
        reader = get_reader()

    regions = detect_text_regions(proc_bgr)
    if not regions:
        regions = [proc_bgr]

    lines = []
    assembled = []
    for reg in regions:
        try:
            res = reader.readtext(reg, detail=1)
            filtered = []
            for bbox, txt, conf in res:
                confv = float(conf) if conf is not None else 0.0
                if confv >= min_confidence:
                    filtered.append({"bbox": bbox, "text": txt, "conf": confv})
            if filtered:
                lines.extend(filtered)
                assembled.append(" ".join([f['text'] for f in filtered]))
        except Exception:
            try:
                single = reader.readtext(reg, detail=0)
                if single:
                    lines.append({"bbox": None, "text": " ".join(single), "conf": None})
                    assembled.append(" ".join(single))
            except Exception:
                continue

    raw_text = " ".join(assembled)
    clean = re.sub(r"\s+", " ", raw_text).strip()

    if detected_lang == 'ur':
        try:
            clean = shape_and_bidi(clean)
            for l in lines:
                if l.get('text'):
                    l['text'] = shape_and_bidi(l['text'])
        except Exception:
            pass

    clean = normalize_digits(clean)

    phones = re.findall(r"\+?\d[\d\s\-]{5,}", clean)
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", clean)
    urls = re.findall(r"(https?://\S+|www\.\S+)", clean)
    amounts = re.findall(r"(?:Rs\.?|INR|\$)\s?\d[\d,]*", clean)

    return {
        "detected_language": detected_lang,
        "raw_text": raw_text,
        "clean_text": clean,
        "lines": lines,
        "extracted_fields": {"phones": phones, "emails": emails, "urls": urls, "amounts": amounts}
    }
