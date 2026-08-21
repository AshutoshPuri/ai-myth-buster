import pickle
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"


# =========================================================
# Device Selection
# =========================================================

def get_device() -> torch.device:
    """Select the best available computation device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# =========================================================
# Load Tokenizer
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}. Train the model before starting inference."
    )

if not LABEL_ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Label encoder not found: {LABEL_ENCODER_PATH}. Train the model before starting inference."
    )

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model architecture: {MODEL_NAME}")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

print(f"Loading trained weights from: {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

print(f"Loading label encoder from: {LABEL_ENCODER_PATH}")
with open(LABEL_ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

print(f"Inference device: {DEVICE}")


# =========================================================
# Prediction Function
# =========================================================

def predict(claim: str) -> dict:
    """Predict whether a claim is a Fact, Myth, or Half-Truth."""
    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty.")

    inputs = tokenizer(
        claim,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities_tensor = torch.softmax(logits, dim=1)
    predicted_class_id = int(torch.argmax(probabilities_tensor, dim=1).item())
    confidence = float(probabilities_tensor[0][predicted_class_id].item())
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]

    probabilities = {}
    for class_id, label in enumerate(label_encoder.classes_):
        probabilities[label] = float(probabilities_tensor[0][class_id].item())

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# Backward-compatible alias used by the existing API layer.
def predict_claim(claim: str):
    result = predict(claim)
    return result["label"], result["confidence"]
