import sys
import os
from paddleocr import PaddleOCR

def run_ocr(image_path: str, lang: str = "ne"):
    """
    Run OCR on the given image using latest PaddleOCR (PP-OCRv5).
    Returns recognized text lines joined by newline, or raises if errors.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Initialize PaddleOCR with default (PP-OCRv5)
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        lang=lang
    )

    # Run OCR
    result = ocr.predict(input=image_path)

    # Print full result (for debug / visualization)
    for res in result:
        # According to docs, each res is a result object
        res.print()
        # If you want to save visual output (image + boxes), you could:
        # res.save_to_img("output")  
        # res.save_to_json("output")
    
    # Extract text lines robustly
    text_lines = []
    for res in result:
        # many result objects store recognized text in `res.rec_texts`
        if hasattr(res, 'rec_texts'):
            text_lines.extend(res.rec_texts)
        # Some versions may store differently — we fallback to checking attributes
        elif hasattr(res, 'texts'):
            text_lines.extend(res.texts)

    return "\n".join(text_lines) if text_lines else ""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_ocr.py /path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    text = run_ocr(img_path, lang="ne")
    print("\n--- OCR TEXT ---")
    print(text)
