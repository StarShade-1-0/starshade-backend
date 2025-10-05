from pydantic import BaseModel

class TessVotingHardPredictResponse(BaseModel):
    prediction: str
    confidence: float | None = None
    class_probabilities: dict[str, float] | None = None
