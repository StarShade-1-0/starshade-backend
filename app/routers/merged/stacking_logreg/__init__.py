from fastapi import APIRouter, HTTPException
from .model_impl import predict as sl_predict
from ..schemas import MergedInferenceFeatures
from .schemas import MergedStackingLogRegPredictResponse

router = APIRouter(prefix="/stacking_logreg")
@router.post("/predict", response_model=MergedStackingLogRegPredictResponse)
def predict(features: MergedInferenceFeatures):
    try:
        pred, conf, class_probs = sl_predict(**features.dict())
        return MergedStackingLogRegPredictResponse(prediction=pred, confidence=conf, class_probabilities=class_probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
