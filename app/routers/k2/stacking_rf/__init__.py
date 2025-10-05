from fastapi import APIRouter, HTTPException
from .model_impl import predict as rf_predict
from ..schemas import K2InferenceFeatures
from .schemas import K2StackingRFPredictResponse

router = APIRouter(prefix="/stacking_rf")

@router.post("/predict", response_model=K2StackingRFPredictResponse)
def predict(features: K2InferenceFeatures):
    try:
        pred, conf, class_probs = rf_predict(**features.dict())
        return K2StackingRFPredictResponse(prediction=pred, confidence=conf, class_probabilities=class_probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
