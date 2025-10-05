from fastapi import APIRouter, HTTPException, UploadFile, File
from .model_impl import predict as vh_predict, batch_predict as vh_batch_predict, FEATURE_COLUMNS
from ..schemas import TessInferenceFeatures
from .schemas import TessVotingHardPredictResponse, TessVotingHardBatchPredictResponse, BatchPredictionResult
import pandas as pd
import io
import numpy as np

router = APIRouter(prefix="/voting_hard")

@router.post("/predict", response_model=TessVotingHardPredictResponse)
def predict(features: TessInferenceFeatures):
    try:
        pred, conf, class_probs = vh_predict(**features.dict())
        return TessVotingHardPredictResponse(prediction=pred, confidence=conf, class_probabilities=class_probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_batch", response_model=TessVotingHardBatchPredictResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction endpoint that accepts a CSV file.
    
    The CSV file should contain columns matching the TESS features:
    - st_pmra, st_pmraerr1, st_pmraerr2, st_pmdec, st_pmdecerr1, st_pmdecerr2
    - pl_tranmid, pl_orbper, pl_trandurh, pl_trandurherr1, pl_trandurherr2
    - pl_trandep, pl_trandeperr1, pl_trandeperr2
    - pl_rade, pl_insol, pl_eqt
    - st_tmag, st_tmagerr1, st_tmagerr2, st_dist, st_disterr1, st_disterr2
    - st_teff, st_tefferr1, st_tefferr2, st_logg, st_rad
    
    Returns predictions, confidence scores, and any errors/warnings.
    """
    warnings = []
    errors = []
    results = []
    
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read CSV file
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        total_rows = len(df)
        
        # Check for required columns
        missing_columns = set(FEATURE_COLUMNS) - set(df.columns)
        extra_columns = set(df.columns) - set(FEATURE_COLUMNS)
        
        if missing_columns:
            warnings.append(f"Missing columns (will be imputed): {', '.join(sorted(missing_columns))}")
        
        if extra_columns:
            warnings.append(f"Extra columns (will be ignored): {', '.join(sorted(extra_columns))}")
        
        # Check for rows with too many missing values
        missing_threshold = 0.8  # Warn if more than 80% of values are missing
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            missing_count = row[list(set(FEATURE_COLUMNS) & set(df.columns))].isna().sum()
            total_expected = len(set(FEATURE_COLUMNS) & set(df.columns))
            if total_expected > 0 and missing_count / total_expected > missing_threshold:
                warnings.append(f"Row {row_idx + 1}: More than 80% of values are missing")
        
        # Perform batch prediction
        try:
            predictions_df = vh_batch_predict(df)
            successful_predictions = 0
            failed_predictions = 0
            
            for row_idx in range(len(predictions_df)):
                try:
                    row = predictions_df.iloc[row_idx]
                    result = BatchPredictionResult(
                        row_number=row_idx + 1,
                        prediction=row['prediction'],
                        confidence=float(row['confidence']) if 'confidence' in row and not pd.isna(row['confidence']) else None,
                        class_probabilities=None,
                        error=None
                    )
                    
                    # Extract class probabilities if available
                    if any(col.startswith('prob_') for col in predictions_df.columns):
                        class_probs = {}
                        for col in predictions_df.columns:
                            if col.startswith('prob_'):
                                class_name = col.replace('prob_', '')
                                prob_value = row[col]
                                if not pd.isna(prob_value):
                                    class_probs[class_name] = float(prob_value)
                        if class_probs:
                            result.class_probabilities = class_probs
                    
                    results.append(result)
                    successful_predictions += 1
                    
                except Exception as e:
                    result = BatchPredictionResult(
                        row_number=row_idx + 1,
                        prediction=None,
                        confidence=None,
                        class_probabilities=None,
                        error=f"Prediction failed: {str(e)}"
                    )
                    results.append(result)
                    failed_predictions += 1
                    errors.append(f"Row {row_idx + 1}: {str(e)}")
            
            return TessVotingHardBatchPredictResponse(
                success=failed_predictions == 0,
                total_rows=total_rows,
                successful_predictions=successful_predictions,
                failed_predictions=failed_predictions,
                warnings=warnings,
                errors=errors,
                results=results
            )
            
        except Exception as e:
            errors.append(f"Batch prediction failed: {str(e)}")
            return TessVotingHardBatchPredictResponse(
                success=False,
                total_rows=total_rows,
                successful_predictions=0,
                failed_predictions=total_rows,
                warnings=warnings,
                errors=errors,
                results=[]
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
