"""
CropGuard ML — FastAPI Backend
REST API for crop yield prediction and disease risk assessment.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    YieldPredictionRequest, YieldPredictionResponse,
    DiseasePredictionRequest, DiseasePredictionResponse,
    DistrictInfo, CropInfo, ModelInfo, HealthResponse,
    FeedbackRequest, FeedbackResponse,
)
from src.predict import CropGuardPredictor
from src.constants import (
    DISTRICT_TO_DIVISION, DIVISIONS, DIVISION_CROPS,
    CROP_SEASON, CROP_MARATHI, CROP_DISEASES, MAJOR_CROPS
)

# ============================================================
# APP INITIALIZATION
# ============================================================
app = FastAPI(
    title="CropGuard ML API",
    description="🌾 AI-powered crop yield prediction & disease risk assessment for Maharashtra",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODEL_DIR = os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models"))
predictor = CropGuardPredictor(model_dir=MODEL_DIR)

# Feedback storage (in-memory for demo; use DB in production)
feedback_log = []


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "CropGuard ML API",
        "description": "Crop yield prediction & disease risk for Maharashtra",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for uptime monitoring."""
    return HealthResponse(
        status="healthy",
        models_loaded=predictor.yield_model is not None and predictor.disease_model is not None,
        version="1.0.0",
    )


@app.post("/predict/yield", response_model=YieldPredictionResponse, tags=["Prediction"])
async def predict_yield(request: YieldPredictionRequest):
    """
    Predict crop yield in kg/hectare.
    Provide district, crop, weather & soil features.
    """
    if request.district not in DISTRICT_TO_DIVISION:
        raise HTTPException(status_code=400,
                            detail=f"Unknown district: {request.district}. "
                                   f"Valid districts: {list(DISTRICT_TO_DIVISION.keys())}")

    input_data = request.model_dump()
    input_data["crop"] = request.crop
    input_data["division"] = DISTRICT_TO_DIVISION.get(request.district, "")

    result = predictor.predict_yield(input_data)

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return YieldPredictionResponse(**result)


@app.post("/predict/disease", response_model=DiseasePredictionResponse, tags=["Prediction"])
async def predict_disease(request: DiseasePredictionRequest):
    """
    Predict disease risk score (0–1) with risk label and recommendations.
    """
    if request.district not in DISTRICT_TO_DIVISION:
        raise HTTPException(status_code=400,
                            detail=f"Unknown district: {request.district}")

    input_data = request.model_dump()
    input_data["crop"] = request.crop
    input_data["division"] = DISTRICT_TO_DIVISION.get(request.district, "")

    result = predictor.predict_disease(input_data)

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return DiseasePredictionResponse(**result)


@app.get("/districts", response_model=list[DistrictInfo], tags=["Information"])
async def get_districts():
    """Returns all 36 Maharashtra districts with division mapping."""
    return [
        DistrictInfo(district=d, division=div)
        for d, div in DISTRICT_TO_DIVISION.items()
    ]


@app.get("/crops/{district}", response_model=list[CropInfo], tags=["Information"])
async def get_crops_for_district(district: str):
    """Returns crops grown in a specific district with season mapping."""
    if district not in DISTRICT_TO_DIVISION:
        raise HTTPException(status_code=404,
                            detail=f"District '{district}' not found")

    division = DISTRICT_TO_DIVISION[district]
    crops = DIVISION_CROPS.get(division, MAJOR_CROPS[:4])

    return [
        CropInfo(
            crop=c,
            season=CROP_SEASON.get(c, "Kharif"),
            crop_marathi=CROP_MARATHI.get(c, c),
        )
        for c in crops
    ]


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
async def get_model_info():
    """Current model version, training date, and accuracy metrics."""
    info = predictor.get_model_info()
    return ModelInfo(**info)


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Farmer submits actual yield for ground-truth logging."""
    feedback_entry = {
        **request.model_dump(),
        "submitted_at": datetime.now().isoformat(),
    }
    feedback_log.append(feedback_entry)

    # Save to file (append mode)
    feedback_file = PROJECT_ROOT / "data" / "feedback_log.json"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(feedback_file.read_text()) if feedback_file.exists() else []
    except Exception:
        existing = []
    existing.append(feedback_entry)
    feedback_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    return FeedbackResponse(
        status="received",
        message=f"Thank you! Feedback for {request.crop} in {request.district} "
                f"({request.season} {request.year}) recorded."
    )


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
