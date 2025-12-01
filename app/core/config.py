import os
from pathlib import Path

# --- BASE PROJECT SETUP ---
PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- MODEL/ASSET PATHS ---
MODEL_DIR = PROJECT_ROOT / "models"

PATH_TO_MODEL = str(MODEL_DIR / "saved_model.pb")
PATH_TO_LABELS = str(MODEL_DIR / "labelmap.pbtxt")

# --- INFERENCE SETTINGS ---
MIN_SCORE = 0.6