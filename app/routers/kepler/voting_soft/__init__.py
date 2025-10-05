from fastapi import APIRouter, HTTPException
from .model_impl import predict as vs_predict
from ..schemas import KeplerInferenceFeatures
from .schemas import KeplerVotingSoftPredictResponse

router = APIRouter(prefix="/voting_soft")
@router.post("/predict", response_model=KeplerVotingSoftPredictResponse)
def predict(features: KeplerInferenceFeatures):
    try:
        pred, conf, class_probs = vs_predict(**features.dict())
        return KeplerVotingSoftPredictResponse(prediction=pred, confidence=conf, class_probabilities=class_probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
