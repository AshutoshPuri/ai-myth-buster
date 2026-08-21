from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "fever"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"

LABELS = ["Fact", "Myth", "Half-Truth"]

FEVER_TO_PROJECT = {
    "SUPPORTS": "Fact",
    "REFUTES": "Myth",
    "NOT ENOUGH INFO": "Half-Truth",
}
