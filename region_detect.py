try:
    import cv2
except Exception:
    cv2 = None
import numpy as np


def _detect_with_cv2(img_bgr, min_area=400, max_area_ratio=0.9):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    blur = cv2.GaussianBlur(grad, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dil = cv2.dilate(th, kernel, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > (w_img * h_img * max_area_ratio):
            continue
        pad_x = int(w * 2 / 100) + 2
        pad_y = int(h * 5 / 100) + 2
        x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x); y1 = min(h_img, y + h + pad_y)
        regions.append((y0, x0, y1, x1))
    regions = sorted(regions, key=lambda r: r[0])
    crops = [img_bgr[y0:y1, x0:x1] for (y0, x0, y1, x1) in regions]
    return crops


def _detect_without_cv2(img_bgr, *_, **__):
    """Fallback when OpenCV is not installed: return empty list so callers fallback to whole image."""
    return []


def detect_text_regions(img_bgr, min_area=400, max_area_ratio=0.9):
    """Detect probable text regions in a BGR OpenCV image and return cropped region images.

    Returns list of numpy arrays (BGR) sorted by vertical position. If OpenCV is not
    available, returns an empty list so the caller will use the whole image.
    """
    if cv2 is not None:
        return _detect_with_cv2(img_bgr, min_area=min_area, max_area_ratio=max_area_ratio)
    else:
        return _detect_without_cv2(img_bgr, min_area=min_area, max_area_ratio=max_area_ratio)
