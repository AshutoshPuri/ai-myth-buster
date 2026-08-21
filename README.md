# AI Myth Buster

A claim-verification system that classifies short news claims as **Fact**, **Myth**, or **Half-Truth**, powered by a fine-tuned BERT model, a FastAPI backend, and a React frontend.

🔗 **Repo:** [github.com/AshutoshPuri/ai-myth-bster](https://github.com/AshutoshPuri/ai-myth-bster.git)

## Overview

Misinformation spreads faster than fact-checkers can keep up with it. This project explores an NLP approach to the problem: given a short claim, classify it as supported by evidence, contradicted by evidence, or unverifiable — served through a real client-server web app rather than a single script.

- **Input:** a natural-language claim (e.g. *"Fox 2000 Pictures released the film Soul Food."*)
- **Output:** one of three labels — `Fact`, `Myth`, `Half-Truth` — with a confidence score and per-class probabilities
- **Model:** `bert-base-uncased`, fine-tuned as a 3-class sequence classifier on the FEVER dataset

## Architecture

```
React Frontend  --HTTP (JSON)-->  FastAPI Backend  -->  BERT Inference Module  -->  model.pt
    (Vite)                          /predict                (model.py)          (fine-tuned checkpoint)
```

The model is served behind a REST API rather than embedded directly in a UI script — this keeps training, inference, and presentation cleanly separated, and lets the frontend be swapped or scaled independently of the ML code.

## Repository Structure

```
ai-myth-buster/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app — /predict, /health routes
│   │   ├── model.py           # Model + tokenizer loading, inference logic
│   │   └── schemas.py         # Pydantic request/response models
│   ├── train/
│   │   ├── train_model.py     # Fine-tuning script
│   │   ├── load_data.py       # JSONL loading + label mapping
│   │   └── evaluate.py        # Metrics on the test split
│   ├── data/fever/            # train.jsonl, dev.jsonl, test.jsonl
│   ├── artifacts/             # model.pt, label_encoder.pkl
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/        # ClaimInput, ResultCard, HistoryList
│   │   ├── api/client.js       # Backend API calls
│   │   └── index.css
│   └── package.json
└── docker-compose.yml
```

## Dataset

Trained and evaluated on [FEVER](https://fever.ai/) (Fact Extraction and VERification), a benchmark of natural-language claims paired with Wikipedia evidence.

| Split | Examples |
|-------|----------|
| train | 19,848 |
| dev   | 9,999 |
| test  | 1,000 |

FEVER's original labels are mapped to this project's target classes:

| FEVER label | Project label |
|-------------|---------------|
| `SUPPORTS` | Fact |
| `REFUTES` | Myth |
| `NOT ENOUGH INFO` | Half-Truth |

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, Hugging Face `transformers` (`bert-base-uncased`) |
| Training | `AdamW`, scikit-learn (label encoding, metrics) |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Frontend | React (Vite), plain CSS |
| Packaging | Docker, docker-compose |

## Setup

**Backend**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

The React app (default `http://localhost:5173`) calls the FastAPI backend at `http://localhost:8000`.

**Training from scratch**
```bash
python backend/train/train_model.py
```
Fine-tunes `bert-base-uncased` for 10 epochs (AdamW, lr=2e-5) and saves `model.pt` + `label_encoder.pkl` to `backend/artifacts/`.

**Docker (full stack)**
```bash
docker compose up --build
```

## Results

Evaluated on the held-out FEVER test split (999 samples):

**Accuracy: 52.65%** — a 3-class task where random guessing would average ~33%.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Fact | 0.4382 | 0.8922 | 0.5878 | 334 |
| Half-Truth | 0.6027 | 0.2667 | 0.3697 | 330 |
| Myth | 0.8092 | 0.4179 | 0.5512 | 335 |
| **Macro avg** | 0.6167 | 0.5256 | 0.5029 | 999 |
| **Weighted avg** | 0.6170 | 0.5265 | 0.5035 | 999 |

**Confusion matrix:**
```
              Fact  Half-Truth  Myth
Fact           298      23       13
Half-Truth     222      88       20
Myth           160      35      140
```

**Reading the results honestly:** the model is strong at catching Myth claims when it commits to that label (80.9% precision) and defaults to "Fact" more than it should — recall on Fact is high (89%) largely because the model over-predicts that class, which drags down Half-Truth and Myth recall. The next clear improvement is addressing this class imbalance in predictions, e.g. via class-weighted loss or more balanced sampling during training.

## Possible Extensions

- Class-weighted training loss to reduce the model's bias toward predicting "Fact"
- Incorporate the FEVER `evidence` field for retrieval-augmented, explainable predictions rather than claim-only classification
- Add authentication/rate-limiting to the API if deployed publicly
- Deploy backend on Hugging Face Spaces or Render, frontend on Vercel/Netlify

## License

No license file is currently included. Add one (e.g. MIT) if you intend for others to reuse this code.
