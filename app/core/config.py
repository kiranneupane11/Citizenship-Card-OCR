from pathlib import Path

# --- BASE PROJECT SETUP ---
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---OUTPUT PATH---
CROPPED_OUTPUT_PATH = PROJECT_ROOT / "assets/cropped-output"

# --- MODEL/ASSET PATHS ---
MODEL_DIR = PROJECT_ROOT / "models"

PATH_TO_MODEL = str(MODEL_DIR)
PATH_TO_LABELS = str(MODEL_DIR / "labelmap.pbtxt")

# --- INFERENCE SETTINGS ---
MIN_SCORE = 0.6
MIN_RESOLUTION = 640