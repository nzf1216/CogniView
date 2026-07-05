from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RETINA_DATA_DIR = DATA_DIR / "retina"
MODEL_DIR = PROJECT_ROOT / "models"
RETINA_MODEL_DIR = MODEL_DIR / "retina"
CHECKPOINT_PATH = (
    RETINA_MODEL_DIR / "best_model.pth"
)
RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_DIR = RESULTS_DIR / "metrics"

PLOTS_DIR = RESULTS_DIR / "plots"

GRADCAM_DIR = RESULTS_DIR / "gradcam"

REPORTS_DIR = RESULTS_DIR / "reports"

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS = 1

LEARNING_RATE = 1e-4

RANDOM_SEED = 42

NUM_CLASSES = 5
MODEL_NAME = "efficientnet_b0"
NUM_WORKERS = 2
PIN_MEMORY = False
RETINA_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)
if __name__ == "__main__":
    print("Project Root:", PROJECT_ROOT)
    print("Retina Data:", RETINA_DATA_DIR)
    print("Checkpoint:", CHECKPOINT_PATH)