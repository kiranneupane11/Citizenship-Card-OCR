import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

import cv2
import numpy as np
from PIL import Image as PILImage

# import config values
from app.core.config import PATH_TO_MODEL, PATH_TO_LABELS, MIN_SCORE


# Global cached network so we don't reload on every call
_net = None
_category_index = None

def load_net(model_path: str):
    """Load the TF .pb model with OpenCV (cached)."""
    global _net
    model_path = str(model_path)
    if _net is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        _net = cv2.dnn.readNetFromTensorflow(model_path)
    return _net

def parse_labelmap(pbtxt_path: str) -> Dict[int, str]:
    """
    Minimal parser for TensorFlow pbtxt labelmap files.
    Returns dict: {id: display_name}
    """
    if not pbtxt_path:
        return {}
    pbtxt_path = str(pbtxt_path)
    if not os.path.exists(pbtxt_path):
        return {}

    category_index = {}
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Very tolerant parsing: find "item { ... }" blocks and extract id & name
    items = content.split("item {")
    for block in items[1:]:
        # find id:
        id_line = [ln for ln in block.splitlines() if "id" in ln]
        name_line = [ln for ln in block.splitlines() if "name" in ln]
        if id_line:
            try:
                id_val = int(id_line[0].split(":")[-1].strip())
            except Exception:
                continue
        else:
            continue

        if name_line:
            raw = name_line[0].split(":", 1)[-1].strip()
            # remove quotes
            name = raw.strip().strip('"').strip("'")
        else:
            name = str(id_val)

        category_index[id_val] = name

    return category_index

def get_best_box_from_detections(
    detections: np.ndarray, image_shape: Tuple[int, int], min_score: float
) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse OpenCV DNN output and return best bounding box (x1,y1,x2,y2) in pixel coords.
    Expected shape for TF Object Detection: (1, 1, N, 7) where each det = [batch, class, score, x1, y1, x2, y2]
    """
    h, w = image_shape
    # handle common shape
    if detections is None:
        return None

    # If detections are (1, 1, N, 7)
    if detections.ndim == 4 and detections.shape[3] == 7:
        best_score = 0.0
        best_box = None
        for det in detections[0, 0, :]:
            score = float(det[2])
            if score >= min_score and score > best_score:
                best_score = score
                x1 = int(det[3] * w)
                y1 = int(det[4] * h)
                x2 = int(det[5] * w)
                y2 = int(det[6] * h)
                best_box = (x1, y1, x2, y2)
        return best_box

    if detections.ndim == 2 and detections.shape[1] >= 4:
        return None

    # fallback: no recognized format
    return None


def crop_id_from_image(input_image):
    """
    Detect and crop the ID card region from an input image and return a PIL.Image (RGB),
    or None if no detection above threshold.
    """

    # --- 1. Normalize input to OpenCV BGR numpy array
    if isinstance(input_image, PILImage.Image):
        # PIL -> RGB numpy -> convert to BGR for OpenCV
        image_rgb = np.array(input_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(input_image, np.ndarray):
        # Assume user supplied a cv2-style BGR array. If it's RGB, convert beforehand.
        image_bgr = input_image.copy()
    else:
        raise TypeError("input_image must be a PIL.Image or a numpy.ndarray")

    h, w = image_bgr.shape[:2]

    # --- 2. Load model and labelmap (cached helpers expected in module)
    net = load_net(PATH_TO_MODEL)

    global category_index
    if category_index is None:
        category_index = parse_labelmap(PATH_TO_LABELS)

    # --- 3. Prepare blob
    blob = cv2.dnn.blobFromImage(image_bgr, scalefactor=1.0, size=(w, h), swapRB=True)
    net.setInput(blob)

    # --- 4. Run inference
    detections = net.forward()

    # --- 5. Parse detections and get best box (in pixel coordinates)
    best_box = get_best_box_from_detections(detections, (h, w), MIN_SCORE)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    # Clip coordinates
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    # --- 6. Crop and return as PIL Image (RGB)
    cropped_bgr = image_bgr[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    cropped_pil = PILImage.fromarray(cropped_rgb)

    return cropped_pil


