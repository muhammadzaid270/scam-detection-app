"""Simple app entrypoint exposing OCR functions.

This module wires the helper functions from the notebook into a single place.
Use `ocr_utils.extract_text_from_image`, `extract_text_whatsapp_aware`, or
`extract_text_with_urdu_support` from your own integrations (e.g. Gemini API).

Example CLI: `python app.py path/to/image.png` will run the Urdu-aware pipeline and print JSON.
"""
import sys
import json
from PIL import Image

# Use top-level imports so `python app.py` works when running from the project directory
from ocr_utils import extract_text_with_urdu_support, extract_text_from_image, extract_text_whatsapp_aware


def run_pipeline(path, mode='urdu'):
    img = Image.open(path)
    if mode == 'simple':
        out = extract_text_from_image(img, engine='easyocr')
    elif mode == 'whatsapp':
        out = extract_text_whatsapp_aware(img, preprocess_func=None)
    else:
        out = extract_text_with_urdu_support(img, preprocess_func=None)
    return out


def display_result(out: dict):
    # Friendly terminal display of important fields
    print('\n--- OCR Result ---')
    clean = out.get('clean_text') or out.get('clean') or out.get('raw_text', '')
    print('\nCLEAN TEXT:\n')
    print(clean[:400] + ('...' if len(clean) > 400 else ''))

    print('\nEXTRACTED FIELDS:')
    ef = out.get('extracted_fields', {})
    for k, v in ef.items():
        print(f" - {k}: {v}")

    lines = out.get('lines', [])
    print(f"\nLINES detected: {len(lines)}")
    if lines:
        print('\nFirst lines:')
        for i, l in enumerate(lines[:6]):
            txt = l.get('text') if isinstance(l, dict) else str(l)
            conf = l.get('conf') if isinstance(l, dict) else None
            print(f" {i+1}. ({conf}) {txt}")

    # Full JSON dump (optional)
    print('\nFull JSON output:')
    print(json.dumps(out, indent=2, ensure_ascii=False))


def choose_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title='Select image file', filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.tiff'), ('All files', '*.*')])
        root.update()
        root.destroy()
        return path
    except Exception:
        return None


def interactive_mode():
    print('Interactive OCR - select image or enter path')
    print('1) Open file dialog')
    print('2) Enter path manually')
    print('3) Exit')
    choice = input('Choose an option [1/2/3]: ').strip()
    if choice == '1':
        path = choose_file_dialog()
        if not path:
            print('No file selected or file dialog unavailable.')
            return
    elif choice == '2':
        path = input('Enter path to image: ').strip('"')
    else:
        print('Exiting.')
        return

    print('\nSelect mode:')
    print(' 1) urdu (default)')
    print(' 2) simple')
    print(' 3) whatsapp')
    m = input('Choose mode [1/2/3]: ').strip()
    mode_map = {'1': 'urdu', '2': 'simple', '3': 'whatsapp'}
    mode = mode_map.get(m, 'urdu')

    try:
        out = run_pipeline(path, mode=mode)
        display_result(out)
    except FileNotFoundError:
        print('File not found:', path)
    except Exception as e:
        print('Error running OCR:', e)


if __name__ == '__main__':
    # If args provided, behave like CLI; otherwise run interactive prompts
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) >= 3 else 'urdu'
        try:
            out = run_pipeline(path, mode=mode)
            display_result(out)
        except Exception as e:
            print('Error:', e)
    else:
        interactive_mode()
