import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(BACKEND_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    from backend.app.config import DATA_DIR, FEVER_TO_PROJECT
except ImportError:
    from app.config import DATA_DIR, FEVER_TO_PROJECT

LABEL_MAP = {
    "SUPPORTS": "Fact",
    "REFUTES": "Myth",
    "NOT ENOUGH INFO": "Half-Truth",
}


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def load_split(path):
    claims = []
    labels = []
    for record in _iter_jsonl(path):
        claim = record.get("claim", "")
        raw_label = record.get("label")
        if raw_label is None:
            continue
        mapped_label = LABEL_MAP.get(raw_label, FEVER_TO_PROJECT.get(raw_label, "Half-Truth"))
        claims.append(claim)
        labels.append(mapped_label)
    return claims, labels


def validate_split(path: str) -> None:
    """Validate a FEVER JSONL split before training."""
    claims, labels = load_split(path)
    print(f"\nFile: {path}")
    print(f"Samples: {len(claims)}")
    if not claims:
        raise ValueError(f"No samples found in {path}")
    if len(claims) != len(labels):
        raise ValueError("Number of claims and labels do not match.")
    print(f"Unique labels: {sorted(set(labels))}")
    distribution = Counter(labels)
    print("\nClass distribution:")
    for label, count in distribution.items():
        print(f"  {label}: {count}")
    print("\nFirst 3 samples:")
    for claim, label in zip(claims[:3], labels[:3]):
        print(f"  [{label}] {claim}")


if __name__ == "__main__":
    validate_split(str(DATA_DIR / "train.jsonl"))
    validate_split(str(DATA_DIR / "dev.jsonl"))
    validate_split(str(DATA_DIR / "test.jsonl"))
