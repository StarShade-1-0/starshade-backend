from pydantic import BaseModel

class MergedStackingLogRegPredictResponse(BaseModel):
    prediction: int
    confidence: float | None = None
    class_probabilities: dict[str, float] | None = None
