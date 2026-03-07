"""
CropGuard ML — Pydantic Schemas for API
Request and response models for all endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# ============================================================
# YIELD PREDICTION
# ============================================================
class YieldPredictionRequest(BaseModel):
    district: str = Field(..., example="Nashik", description="Maharashtra district name")
    crop: str = Field(..., example="Onion", description="Crop name")
    season: str = Field("Rabi", example="Rabi", description="Kharif / Rabi / Summer / Year-Round")
    year: int = Field(2025, example=2025)
    rainfall_mm_seasonal: float = Field(..., example=420.5, description="Seasonal rainfall in mm")
    soil_pH: float = Field(7.0, example=7.1, ge=3.0, le=11.0)
    irrigation_type: str = Field("rainfed", example="drip")
    fertilizer_kg_ha: float = Field(150.0, example=180.0, ge=0)
    sowing_date_deviation: int = Field(0, example=-3, description="Days early(-) or late(+)")
    ndvi_peak: float = Field(0.6, example=0.72, ge=0.0, le=1.0)
    humidity_avg_pct: Optional[float] = Field(60.0, ge=0, le=100)
    temp_max_avg_C: Optional[float] = Field(32.0)
    temp_min_avg_C: Optional[float] = Field(20.0)
    organic_carbon_pct: Optional[float] = Field(0.5, ge=0)
    nitrogen_kg_ha: Optional[float] = Field(200.0, ge=0)
    soil_moisture_pct: Optional[float] = Field(25.0, ge=0, le=100)


class YieldPredictionResponse(BaseModel):
    predicted_yield_kg_ha: float
    confidence_interval: List[float]
    top_yield_drivers: List[str]
    model_version: str
    district_avg_yield: float


# ============================================================
# DISEASE RISK
# ============================================================
class DiseasePredictionRequest(BaseModel):
    district: str = Field(..., example="Yavatmal", description="Maharashtra district name")
    crop: str = Field(..., example="Cotton", description="Crop name")
    humidity_avg_pct: float = Field(..., example=82.0, ge=0, le=100)
    temp_max_avg_C: float = Field(..., example=33.5)
    humidity_streak_days: int = Field(0, example=8, ge=0)
    temp_disease_window: int = Field(0, example=12, ge=0)
    ndvi_peak: float = Field(0.6, example=0.55, ge=0.0, le=1.0)
    ndvi_stress_weeks: int = Field(0, example=2, ge=0)
    prev_season_disease: int = Field(0, example=1, ge=0, le=1)
    pest_alert_issued: int = Field(0, example=0, ge=0, le=1)
    rainfall_mm_seasonal: Optional[float] = Field(600.0)
    soil_moisture_pct: Optional[float] = Field(25.0)


class DiseasePredictionResponse(BaseModel):
    risk_score: float
    risk_label: str
    risk_label_marathi: str
    top_risk_factors: List[str]
    probable_diseases: List[str]
    recommendations: List[str]
    crop_marathi: str
    model_version: str


# ============================================================
# INFO ENDPOINTS
# ============================================================
class DistrictInfo(BaseModel):
    district: str
    division: str


class CropInfo(BaseModel):
    crop: str
    season: str
    crop_marathi: str


class ModelInfo(BaseModel):
    yield_model: dict
    disease_model: dict
    version: str
    region: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    models_loaded: bool
    version: str


# ============================================================
# FEEDBACK
# ============================================================
class FeedbackRequest(BaseModel):
    district: str
    crop: str
    season: str
    year: int
    actual_yield_kg_ha: Optional[float] = None
    actual_disease_observed: Optional[str] = None
    farmer_name: Optional[str] = None
    comments: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str = "received"
    message: str
