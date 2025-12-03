import cv2
import os
import sys
import json
import traceback
import multiprocessing as mp
import io

from app.utils import model_inference, preprocessing
from app.ocr import run_ocr
from app.core.config import MIN_RESOLUTION


def process_id_card(image_path_or_array):
    # 1. Load Image
    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            raise ValueError(f"Image path does not exist: {image_path_or_array}")
        original_image = cv2.imread(image_path_or_array)
        if original_image is None:
            raise ValueError(f"cv2.imread failed to load image: {image_path_or_array}")
    else:
        # assume it's already an image array
        original_image = image_path_or_array

    if original_image is not None:
        try:
            h, w = original_image.shape[:2]
            print(f"Original Image shape: {w}x{h}")
            if max(w, h) < MIN_RESOLUTION:
                raise ValueError(
                    f"Image quality too low! Please provide a higher-resolution image of minimum {MIN_RESOLUTION} resolution."
                )
        except AttributeError:
            print("loaded image (no shape available or array is malformed)")


    # 2. INFERENCE LAYER: Find and Crop ID
    cropped_id = model_inference.detect_card(original_image)

    if cropped_id is None:
        raise ValueError("No ID card detected in the image.")
    
    try:
        ch, cw = cropped_id.shape[:2]
        print(f"[debug] cropped id shape: {cw}x{ch}")
    except Exception:
        pass

    # 3. PREPROCESSING LAYER: Skew, Resize, Border
    processed_image = preprocessing.preprocess_pipeline(cropped_id)
    if processed_image is None:
        raise ValueError("preprocessing.preprocess_pipeline returned None")
    
    

    # 4. OCR LAYER: Extract Text
    text = run_ocr.run_ocr(processed_image)

    return {
        "status": "success",
        "raw_text": text,
        # You could also return structured data if you parse the text later
    }

def main(argv):
    if len(argv) < 2:
        print("Usage: python -m app.service.ocr_service <image_path>")
        sys.exit(2)

    image_path = argv[1]

    try:
        result = process_id_card(image_path)
    except Exception as e:
        # structured error output and full traceback for debugging
        error_output = {
            "status": "error",
            "message": str(e),
            "type": e.__class__.__name__
        }
        print(json.dumps(error_output, ensure_ascii=False, indent=2))
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)