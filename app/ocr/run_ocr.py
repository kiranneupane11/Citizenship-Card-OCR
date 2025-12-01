import os
import cv2
from paddleocr import PaddleOCR

def run_ocr(image_path: str):
    """Runs OCR on the given image and prints recognized text."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Initialize OCR only once
    ocr = PaddleOCR(use_textline_orientation=True, lang='ne')  # Nepali language

    # --- Run OCR ---
    result = ocr.predict(image_path)

    # Extract text lines
    text_lines = []
    for res in result:
        if isinstance(res, dict) and 'rec_texts' in res:
            text_lines.extend(res['rec_texts'])

    # Return text
    return "\n".join(text_lines) if text_lines else "No text found"

