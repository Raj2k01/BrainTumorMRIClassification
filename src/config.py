import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "Epic and CSCR hospital Dataset"
)

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BASELINE_MODEL_PATH = (
    OUTPUT_DIR
    / "best_brain_tumor_cnn.pth"
)

TUNED_MODEL_PATH = (
    OUTPUT_DIR
    / "best_improved_tuned_cnn.pth"
)

PRODUCTION_MODEL_PATH = (
    OUTPUT_DIR
    / "best_production_brain_tumor_cnn.pth"
)

FINAL_MODEL_PATH = TUNED_MODEL_PATH

MODEL_PATH = PRODUCTION_MODEL_PATH


TRAINING_HISTORY_PATH = (
    OUTPUT_DIR
    / "production_training_history.csv"
)

HISTORY_PATH = TRAINING_HISTORY_PATH

CLASSIFICATION_REPORT_PATH = (
    OUTPUT_DIR
    / "classification_report.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)

TEST_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "test_predictions.csv"
)


IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.40

VALIDATION_SIZE = 0.15
EARLY_STOPPING_PATIENCE = 5
CONFIDENCE_THRESHOLD = 0.70

RANDOM_SEED = 42

NUM_WORKERS = 0
PIN_MEMORY = False


MAX_UPLOAD_SIZE_MB = 10

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB
    * 1024
    * 1024
)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


APP_NAME = "Brain Tumor MRI Classification API"
APP_VERSION = "1.0.0"

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development"
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        (
            "http://localhost:5173,"
            "http://127.0.0.1:5173"
        )
    ).split(",")
    if origin.strip()
]