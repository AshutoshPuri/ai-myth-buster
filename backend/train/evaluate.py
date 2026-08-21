import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
TRAIN_ROOT = BACKEND_ROOT / "train"
for entry in (str(PROJECT_ROOT), str(BACKEND_ROOT), str(TRAIN_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    from backend.app.config import ARTIFACTS_DIR, DATA_DIR
    from backend.train.load_data import load_split
    from backend.train.train_model import ClaimDataset
except ModuleNotFoundError:
    from app.config import ARTIFACTS_DIR, DATA_DIR
    from train.load_data import load_split
    from train.train_model import ClaimDataset


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = PROJECT_ROOT / "backend" / "data" / "fever" / "test.jsonl"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
EVAL_DIR = ARTIFACTS_DIR / "eval"
REPORT_PATH = EVAL_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = EVAL_DIR / "confusion_matrix.png"


def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_claims(tokenizer, claims):
    return tokenizer(
        claims,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def evaluate_model(batch_size: int):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"\nUsing device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\nTrain the model before running evaluation."
        )

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {LABEL_ENCODER_PATH}\nTrain the model before running evaluation."
        )

    with open(LABEL_ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)

    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    print("\nLoading test data...")
    claims, labels = load_split(str(TEST_DATA_PATH))
    print(f"Test samples: {len(claims)}")

    encoded_labels = label_encoder.transform(labels)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    encodings = tokenize_claims(tokenizer, claims)

    dataset = ClaimDataset(encodings, encoded_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\nLoading model: {MODEL_NAME}")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    predictions = []
    actual_labels = []

    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch in data_loader:
            labels_batch = batch["labels"]
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            actual_labels.extend(labels_batch.tolist())

    accuracy = accuracy_score(actual_labels, predictions)
    report = classification_report(actual_labels, predictions, target_names=class_names, digits=4)
    matrix = confusion_matrix(actual_labels, predictions)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:\n")
    print(matrix)

    report_text = (
        f"AI Myth Buster Evaluation\n"
        f"{'=' * 60}\n\n"
        f"Model: {MODEL_NAME}\n"
        f"Test samples: {len(claims)}\n\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Classification Report:\n\n"
        f"{report}\n"
        f"Confusion Matrix:\n\n"
        f"{matrix}\n"
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Report saved to: {REPORT_PATH}")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("AI Myth Buster - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the AI Myth Buster BERT classifier.")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size (default: 32).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_model(batch_size=args.batch_size)
