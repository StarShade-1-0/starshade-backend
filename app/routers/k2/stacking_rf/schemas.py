from pydantic import BaseModel

class K2StackingRFPredictResponse(BaseModel):
    prediction: str
    confidence: float = None
    class_probabilities: dict[str, float] = None
