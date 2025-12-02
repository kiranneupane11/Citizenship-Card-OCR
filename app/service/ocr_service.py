import cv2
import numpy as np

# Import your modules
from app.utils import model_inference, preprocessing
from app.ocr import run_ocr

def process_id_card(image_path_or_array):
    # 1. Load Image
    if isinstance(image_path_or_array, str):
        original_image = cv2.imread(image_path_or_array)
    else:
        original_image = image_path_or_array

    # 2. INFERENCE LAYER: Find and Crop ID
    cropped_id = model_inference.crop_id_from_image(original_image)

    if cropped_id is None:
        raise ValueError("No ID card detected in the image.")

    # 3. PREPROCESSING LAYER: Skew, Resize, Border
    processed_image = preprocessing.preprocess_pipeline(cropped_id)

    # 4. OCR LAYER: Extract Text
    text = run_ocr.run_ocr(processed_image)

    return {
        "status": "success",
        "raw_text": text,
        # You could also return structured data if you parse the text later
    }