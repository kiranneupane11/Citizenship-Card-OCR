import sys
from paddleocr import PaddleOCR

def run_ocr(image, lang: str="ne"):
    """
    Run OCR on the given image using latest PaddleOCR (PP-OCRv5).
    Returns recognized text lines joined by newline, or raises if errors.
    """
    if image is None:
        raise FileNotFoundError(f"Image {image} not found")

    # Initialize PaddleOCR with default (PP-OCRv5)
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        lang=lang
    )

    # Run OCR
    result = ocr.predict(input=image)

    # Print full result
    for res in result:
        # each res is a result object
        res.print()
    # Extract text lines robustly
    text_lines = []
    for res in result:
        if isinstance(res, dict) and 'rec_texts' in res:
            text_lines.extend(res['rec_texts'])
    text = "\n".join(text_lines) if text_lines else "No text found"

    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_ocr.py /path/to/image.jpg")
        sys.exit(1)

    processed_image = sys.argv[1]
    text = run_ocr(processed_image, lang="ne")
    print("\n--- OCR TEXT ---")
    print(text)
