import arabic_reshaper
from bidi.algorithm import get_display

ARABIC_INDIC_DIGITS = {
    ord('٠'): '0', ord('١'): '1', ord('٢'): '2', ord('٣'): '3', ord('٤'): '4',
    ord('٥'): '5', ord('٦'): '6', ord('٧'): '7', ord('٨'): '8', ord('٩'): '9',
    ord('۰'): '0', ord('۱'): '1', ord('۲'): '2', ord('۳'): '3', ord('۴'): '4',
    ord('۵'): '5', ord('۶'): '6', ord('۷'): '7', ord('۸'): '8', ord('۹'): '9',
}

def normalize_digits(text):
    return text.translate(ARABIC_INDIC_DIGITS)

def shape_and_bidi(urdu_text):
    """Reshape Urdu/Arabic script and apply bidi for correct visual order."""
    reshaped = arabic_reshaper.reshape(urdu_text)
    bidi_text = get_display(reshaped)
    return normalize_digits(bidi_text)
