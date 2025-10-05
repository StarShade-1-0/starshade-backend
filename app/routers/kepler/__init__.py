from fastapi import APIRouter, Response
from fastapi.responses import FileResponse
import os
from .voting_soft import router as voting_soft_router

router = APIRouter(prefix="/kepler", tags=["Kepler"])

@router.get("/dataset")
def get_kepler_dataset():
    # Get the absolute path to the dataset file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "dataset.csv")

    # Check if file exists
    if os.path.exists(dataset_path):
        return FileResponse(
            path=dataset_path,
            filename="dataset.csv",
            media_type="text/csv"
        )
    else:
        return Response(content="Dataset file not found", status_code=404)

router.include_router(voting_soft_router)
