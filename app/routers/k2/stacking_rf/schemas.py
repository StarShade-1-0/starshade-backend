from pydantic import BaseModel
from typing import List, Dict, Any

class K2StackingRFPredictResponse(BaseModel):
    prediction: str
    confidence: float | None = None
    class_probabilities: dict[str, float] | None = None

class BatchPredictionResult(BaseModel):
    row_number: int
    prediction: str | None = None
    confidence: float | None = None
    class_probabilities: dict[str, float] | None = None
    error: str | None = None

class K2StackingRFBatchPredictResponse(BaseModel):
    success: bool
    total_rows: int
    successful_predictions: int
    failed_predictions: int
    warnings: List[str] = []
    errors: List[str] = []
    results: List[BatchPredictionResult] = []
