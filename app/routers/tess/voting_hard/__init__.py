from fastapi import APIRouter, HTTPException
from .model_impl import predict as vh_predict
from ..schemas import TessInferenceFeatures
from .schemas import TessVotingHardPredictResponse

router = APIRouter(prefix="/voting_hard")

@router.post("/predict", response_model=TessVotingHardPredictResponse)
def predict(features: TessInferenceFeatures):
    try:
        pred, conf, class_probs = vh_predict(**features.dict())
        return TessVotingHardPredictResponse(prediction=pred, confidence=conf, class_probabilities=class_probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
