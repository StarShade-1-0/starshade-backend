from pydantic import BaseModel
from typing import List, Dict, Any

class MergedStackingLogRegPredictResponse(BaseModel):
    prediction: int
    confidence: float | None = None
    class_probabilities: dict[str, float] | None = None

class BatchPredictionResult(BaseModel):
    row_number: int
    prediction: int | None = None
    confidence: float | None = None
    class_probabilities: dict[str, float] | None = None
    error: str | None = None

class MergedStackingLogRegBatchPredictResponse(BaseModel):
    success: bool
    total_rows: int
    successful_predictions: int
    failed_predictions: int
    warnings: List[str] = []
    errors: List[str] = []
    results: List[BatchPredictionResult] = []
