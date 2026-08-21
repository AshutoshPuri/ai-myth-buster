from pydantic import BaseModel, Field


class ClaimRequest(BaseModel):
    """Request body for claim prediction."""

    claim: str = Field(
        ...,
        min_length=1,
        description="Natural-language claim to classify.",
        examples=["The Earth revolves around the Sun."],
    )


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    label: str = Field(
        ...,
        description="Predicted class: Fact, Myth, or Half-Truth.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1.",
    )

    probabilities: dict[str, float] = Field(
        ...,
        description="Probability assigned to each class.",
    )
