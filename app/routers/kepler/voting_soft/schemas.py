from pydantic import BaseModel

class KeplerVotingSoftPredictResponse(BaseModel):
    prediction: str
    confidence: float | None = None
    class_probabilities: dict[str, float] | None = None
