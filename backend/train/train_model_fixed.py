import argparse
import pickle
import sys
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
TRAIN_ROOT = BACKEND_ROOT / "train"
for entry in (str(PROJECT_ROOT), str(BACKEND_ROOT), str(TRAIN_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from backend.app.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
)
from backend.train.load_data import load_split

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

TRAIN_DATA_PATH = DATA_DIR / "train.jsonl"
DEV_DATA_PATH = DATA_DIR / "dev.jsonl"


class ClaimDataset(Dataset):
    """PyTorch dataset for tokenized claim classification data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for AI Myth Buster.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training and validation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit the number of training examples used.",
    )
    parser.add_argument(
        "--max_dev_samples",
        type=int,
        default=None,
        help="Limit the number of validation examples used.",
    )
    return parser.parse_args()


def get_device():
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


def validate(model, data_loader, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(epochs: int, batch_size: int, learning_rate: float, max_train_samples: int | None = None, max_dev_samples: int | None = None):
    device = get_device()
    print(f"\nUsing device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading training data...")
    train_claims, train_labels = load_split(str(TRAIN_DATA_PATH))
    if max_train_samples is not None:
        train_claims = train_claims[:max_train_samples]
        train_labels = train_labels[:max_train_samples]
    print(f"Training samples: {len(train_claims)}")

    print("\nLoading validation data...")
    dev_claims, dev_labels = load_split(str(DEV_DATA_PATH))
    if max_dev_samples is not None:
        dev_claims = dev_claims[:max_dev_samples]
        dev_labels = dev_labels[:max_dev_samples]
    print(f"Validation samples: {len(dev_claims)}")

    label_encoder = LabelEncoder()
    train_encoded_labels = label_encoder.fit_transform(train_labels)
    dev_encoded_labels = label_encoder.transform(dev_labels)

    print("\nLabel mapping:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {index}")

    with open(LABEL_ENCODER_PATH, "wb") as file:
        pickle.dump(label_encoder, file)

    print(f"\nLabel encoder saved to: {LABEL_ENCODER_PATH}")

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_encodings = tokenize_claims(tokenizer, train_claims)
    dev_encodings = tokenize_claims(tokenizer, dev_claims)

    train_dataset = ClaimDataset(train_encodings, train_encoded_labels)
    dev_dataset = ClaimDataset(dev_encodings, dev_encoded_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nLoading model: {MODEL_NAME}")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("\nStarting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    for epoch in range(epochs):
        model.train()
        total_training_loss = 0.0

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 60}")

        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

            if (batch_index + 1) % 100 == 0:
                print(f"Batch {batch_index + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        average_training_loss = total_training_loss / len(train_loader)
        validation_loss, validation_accuracy = validate(model, dev_loader, device)

        print(f"\nTraining Loss: {average_training_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    args = parse_arguments()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
    )
