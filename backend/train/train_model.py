import argparse
import pickle

import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

from backend.app.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
)
from backend.train.load_data import load_split

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


class ClaimDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_claims(tokenizer, claims):
    return tokenizer(claims, padding="max_length", truncation=True, max_length=MAX_LENGTH)


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

    return total_loss / len(data_loader), correct / total


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    return parser.parse_args()


def train_model(epochs: int, batch_size: int, learning_rate: float):
    device = get_device()
    print(f"Using device: {device}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_claims, train_labels = load_split(str(DATA_DIR / "train.jsonl"))
    dev_claims, dev_labels = load_split(str(DATA_DIR / "dev.jsonl"))

    label_encoder = LabelEncoder()
    train_encoded_labels = label_encoder.fit_transform(train_labels)
    dev_encoded_labels = label_encoder.transform(dev_labels)

    with open(LABEL_ENCODER_PATH, "wb") as file:
        pickle.dump(label_encoder, file)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenize_claims(tokenizer, train_claims)
    dev_encodings = tokenize_claims(tokenizer, dev_claims)

    train_dataset = ClaimDataset(train_encodings, train_encoded_labels)
    dev_dataset = ClaimDataset(dev_encodings, dev_encoded_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        validation_loss, validation_accuracy = validate(model, dev_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={total_loss / len(train_loader):.4f}, val_loss={validation_loss:.4f}, val_acc={validation_accuracy:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    args = parse_arguments()
    train_model(args.epochs, args.batch_size, args.learning_rate)
