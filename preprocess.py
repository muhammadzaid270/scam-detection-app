try:
    import cv2
except Exception:
    cv2 = None
import numpy as np


def _preprocess_with_cv2(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    h, w = gray.shape
    if min(h, w) < 400:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return th


def _preprocess_without_cv2(img_bgr):
    """Fallback preprocessing using numpy/PIL when OpenCV is not available.

    This provides a lightweight grayscale + simple threshold to allow OCR to run.
    """
    arr = np.array(img_bgr)
    # if image is PIL Image, convert
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # assume RGB or BGR — convert to grayscale using luminosity
        r = arr[..., 0].astype(np.float32)
        g = arr[..., 1].astype(np.float32)
        b = arr[..., 2].astype(np.float32)
        # approximate luminance (works for RGB or BGR ordering for OCR purposes)
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
    else:
        gray = arr.astype(np.uint8)

    h, w = gray.shape[:2]
    if min(h, w) < 400:
        # upscale by simple nearest-neighbor
        gray = np.repeat(np.repeat(gray, 2, axis=0), 2, axis=1)

    # simple global threshold at mean intensity
    thresh = int(gray.mean())
    th = (gray > thresh).astype(np.uint8) * 255
    return th


def preprocess_image(img_bgr):
    """Public preprocessing function. Uses OpenCV when available, otherwise a numpy fallback."""
    if cv2 is not None:
        return _preprocess_with_cv2(img_bgr)
    else:
        return _preprocess_without_cv2(img_bgr)

