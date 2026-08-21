from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.model import predict
    from app.schemas import ClaimRequest, PredictionResponse
except ModuleNotFoundError:
    from backend.app.model import predict
    from backend.app.schemas import ClaimRequest, PredictionResponse


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="AI Myth Buster API",
    description=(
        "REST API for classifying claims as "
        "Fact, Myth, or Half-Truth using BERT."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Health Check
# =========================================================

@app.get(
    "/health",
    tags=["Health"],
)
def health_check() -> dict[str, str]:
    """Check whether the API is running."""
    return {"status": "ok"}


# =========================================================
# Prediction
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict_claim(
    request: ClaimRequest,
) -> PredictionResponse:
    """Classify a natural-language claim."""
    try:
        result = predict(request.claim)
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        print(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from exc


# =========================================================
# Run Command
# =========================================================
# uvicorn app.main:app --reload --port 8000
