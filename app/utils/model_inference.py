import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
import argparse

from app.core.config import PATH_TO_MODEL,PATH_TO_LABELS, MIN_SCORE

CROPPED_OUTPUT_PATH = 'assets/cropped-output'

def parse_labelmap(labelmap_path=PATH_TO_LABELS):
    """
    Parses the labelmap.pbtxt file to create a category index.
    """
    classes = {}
    current_id = None
    current_name = None
    if not os.path.exists(labelmap_path):
        return {1: {'id': 1, 'name': 'object'}}  # Fallback

    with open(labelmap_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id:'):
                try:
                    current_id = int(line.split(':')[1].strip())
                except:
                    current_id = None
            elif line.startswith('display_name:'):
                current_name = line.split(':', 1)[1].strip().strip('"').strip("'")
            elif line == '}':
                if current_id and current_name:
                    classes[current_id] = {'id': current_id, 'name': current_name}
                current_id, current_name = None, None
    return classes if classes else {1: {'id': 1, 'name': 'object'}}

def load_model(model_path=PATH_TO_MODEL):
    """
    Loads the TensorFlow SavedModel for object detection.
    """
    try:
        detect_module = tf.saved_model.load(model_path)
        detect_fn = detect_module.signatures.get('serving_default', next(iter(detect_module.signatures.values())))
        print("   -> Model loaded successfully.")
        return detect_fn
    except Exception as e:
        raise RuntimeError(f"Could not load model at {model_path}. Check paths. Error: {e}")

def load_image(image_path):
    """
    Loads and preprocesses the image into a TensorFlow tensor.
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Could not read image file: {image_path}")
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_cv, axis=0), dtype=tf.uint8)
    return image_cv, input_tensor

def run_detection(detect_fn, input_tensor):
    """
    Runs object detection on the input tensor and returns detection outputs.
    """
    # Get input signature key (usually 'input_tensor')
    _, sig_kwargs = detect_fn.structured_input_signature
    input_key = list(sig_kwargs.keys())[0]
    outputs = detect_fn(**{input_key: input_tensor})
    boxes = outputs['detection_boxes'][0].numpy()
    scores = outputs['detection_scores'][0].numpy()
    classes = outputs['detection_classes'][0].numpy().astype(np.int32)
    return boxes, scores, classes

def get_crop_coordinates(scores, boxes, classes, category_index, min_score):
    """
    Analyzes detection results and returns the box coordinates [ymin, xmin, ymax, xmax]
    for the highest scoring object.
    """
    best_idx = None
    if scores is not None and len(scores) > 0:
        best_idx = int(np.argmax(scores))

    if best_idx is not None and (scores[best_idx] >= min_score):
        ymin, xmin, ymax, xmax = boxes[best_idx]
        cls_id = int(classes[best_idx])
        cls_name = category_index.get(cls_id, {'name': 'object'})['name']
        print(f"   -> Detection Found: {cls_name} ({int(scores[best_idx]*100)}%)")
        return [ymin, xmin, ymax, xmax]

    print("   -> No object exceeded the confidence threshold.")
    return [0.0, 0.0, 1.0, 1.0]

def crop_image(image_cv, ymin, xmin, ymax, xmax):
    """
    Crops the image based on normalized coordinates and returns the cropped PIL image.
    """
    im_height, im_width, _ = image_cv.shape
    left = int(xmin * im_width)
    right = int(xmax * im_width)
    top = int(ymin * im_height)
    bottom = int(ymax * im_height)
    # Convert CV2 (BGR) to PIL (RGB)
    image_pil = PILImage.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    roi_box = (max(0, left), max(0, top), min(im_width, right), min(im_height, bottom))
    cropped_im = image_pil.crop(roi_box)
    return cropped_im

def process_image(image_path):

    print(f"\n--- Processing: {image_path} ---")

    # Load model
    print("1. Loading TF2 SavedModel...")
    detect_fn = load_model(PATH_TO_MODEL)

    # Parse labelmap
    category_index = parse_labelmap(PATH_TO_LABELS)

    # Load image
    print("2. Preparing image...")
    image_cv, input_tensor = load_image(image_path)

    # Run detection
    print("3. Running detection...")
    boxes, scores, classes = run_detection(detect_fn, input_tensor)

    # Get crop coordinates
    print("4. Analyzing detections...")
    ymin, xmin, ymax, xmax = get_crop_coordinates(scores, boxes, classes, category_index, MIN_SCORE)

    # Crop image
    cropped = crop_image(image_cv, ymin, xmin, ymax, xmax)

    return cropped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop ID card using trained TF2 model")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default=CROPPED_OUTPUT_PATH,
                        help="Output path for cropped image")
    args = parser.parse_args()

    process_image(args.image_path)