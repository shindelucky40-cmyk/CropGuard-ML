"""
CropGuard ML — Prediction / Inference Module
Loads trained models and returns predictions with explanations.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.constants import (
    DISTRICT_TO_DIVISION, CROP_AVG_YIELD, CROP_WATER_REQUIREMENT,
    IRRIGATION_EFFICIENCY, CROP_DISEASES, RISK_THRESHOLDS,
    RISK_LABELS_MARATHI, CROP_MARATHI
)


class CropGuardPredictor:
    """Unified predictor for yield and disease risk."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self._load_models()

    def _load_models(self):
        """Load yield and disease models + metadata."""
        # Yield model
        yield_path = self.model_dir / "yield_model.joblib"
        yield_meta_path = self.model_dir / "yield_model_metadata.json"
        if yield_path.exists():
            self.yield_model = joblib.load(yield_path)
            with open(yield_meta_path) as f:
                self.yield_meta = json.load(f)
        else:
            self.yield_model = None
            self.yield_meta = {}

        # Disease model
        disease_path = self.model_dir / "disease_model.joblib"
        disease_meta_path = self.model_dir / "disease_model_metadata.json"
        if disease_path.exists():
            self.disease_model = joblib.load(disease_path)
            with open(disease_meta_path) as f:
                self.disease_meta = json.load(f)
            # Load label encoder
            le_path = self.model_dir / "disease_label_encoder.joblib"
            if le_path.exists():
                self.disease_le = joblib.load(le_path)
            else:
                self.disease_le = None
        else:
            self.disease_model = None
            self.disease_meta = {}
            self.disease_le = None

    def predict_yield(self, input_data: dict) -> dict:
        """
        Predict crop yield.
        Returns: predicted_yield, confidence_interval, top_drivers, model_version
        """
        if self.yield_model is None:
            return {"error": "Yield model not loaded"}

        feature_cols = self.yield_meta.get("feature_columns", [])
        row = self._prepare_features(input_data, feature_cols)

        prediction = float(self.yield_model.predict(row)[0])

        # Confidence interval (±15% of prediction as baseline)
        ci_low = round(prediction * 0.85, 1)
        ci_high = round(prediction * 1.15, 1)

        # Top yield drivers from model metadata
        top_drivers = self.yield_meta.get("top_features", [])[:5]

        district_avg = CROP_AVG_YIELD.get(input_data.get("crop", ""), 1500)

        return {
            "predicted_yield_kg_ha": round(prediction, 1),
            "confidence_interval": [ci_low, ci_high],
            "top_yield_drivers": top_drivers,
            "model_version": "v1.0.0",
            "district_avg_yield": district_avg,
        }

    def predict_disease(self, input_data: dict) -> dict:
        """
        Predict disease risk.
        Returns: risk_score, risk_label, risk_label_marathi, top_factors, recommendations
        """
        if self.disease_model is None:
            return {"error": "Disease model not loaded"}

        feature_cols = self.disease_meta.get("feature_columns", [])
        row = self._prepare_features(input_data, feature_cols)

        # Get probability predictions
        proba = self.disease_model.predict_proba(row)[0]
        pred_class = int(np.argmax(proba))

        label_order = self.disease_meta.get("label_order", ["Low", "Medium", "High", "Critical"])
        risk_label = label_order[pred_class]

        # Risk score = weighted probability (higher classes contribute more)
        risk_score = float(np.sum(proba * np.array([0.125, 0.375, 0.625, 0.875])))
        risk_score = round(min(1.0, max(0.0, risk_score)), 3)

        # Top risk factors
        top_factors = self.disease_meta.get("top_features", [])[:3]

        crop = input_data.get("crop", "")
        diseases = CROP_DISEASES.get(crop, [])

        # Recommendations based on risk level
        recommendations = self._get_recommendations(risk_label, crop)

        return {
            "risk_score": risk_score,
            "risk_label": risk_label,
            "risk_label_marathi": RISK_LABELS_MARATHI.get(risk_label, ""),
            "top_risk_factors": top_factors,
            "probable_diseases": diseases[:3],
            "recommendations": recommendations,
            "crop_marathi": CROP_MARATHI.get(crop, crop),
            "model_version": "v1.0.0",
        }

    def _prepare_features(self, input_data: dict, feature_cols: list) -> pd.DataFrame:
        """Convert input dict to DataFrame with correct feature columns."""
        # Map input to data row
        row_data = {}
        for col in feature_cols:
            if col in input_data:
                row_data[col] = input_data[col]
            else:
                row_data[col] = 0

        df = pd.DataFrame([row_data])
        return df[feature_cols]

    def _get_recommendations(self, risk_label: str, crop: str) -> list:
        """Get actionable recommendations based on risk level."""
        recs = {
            "Low": [
                "Continue regular monitoring",
                "Maintain current crop management practices",
                f"नियमित निरीक्षण सुरू ठेवा (Continue monitoring)"
            ],
            "Medium": [
                "Increase field scouting frequency to every 3 days",
                "Consider preventive fungicide application",
                "Monitor weather forecast for extended humidity",
                f"प्रतिबंधात्मक उपाय करा (Take preventive measures)"
            ],
            "High": [
                "Immediately apply recommended fungicide/pesticide",
                "Consult nearest KVK (Krishi Vigyan Kendra)",
                "Set up pheromone traps if pest-related",
                "Reduce irrigation to lower field humidity",
                f"तात्काळ कृषी विज्ञान केंद्राशी संपर्क साधा (Contact KVK immediately)"
            ],
            "Critical": [
                "URGENT: Apply curative treatment within 48 hours",
                "Contact District Agriculture Officer immediately",
                "Consider harvesting early to minimize losses",
                "Report to Crop Insurance provider",
                "Document damage for government relief applications",
                f"तात्काळ उपचार करा! जिल्हा कृषी अधिकाऱ्यांशी संपर्क करा (Urgent treatment needed!)"
            ],
        }
        return recs.get(risk_label, recs["Low"])

    def get_model_info(self) -> dict:
        """Return current model information."""
        return {
            "yield_model": {
                "type": self.yield_meta.get("model_type", "Not loaded"),
                "metrics": self.yield_meta.get("metrics", {}).get("test", {}),
                "n_features": len(self.yield_meta.get("feature_columns", [])),
                "n_train_samples": self.yield_meta.get("n_train_samples", 0),
            },
            "disease_model": {
                "type": self.disease_meta.get("model_type", "Not loaded"),
                "metrics": self.disease_meta.get("metrics", {}).get("test", {}),
                "n_features": len(self.disease_meta.get("feature_columns", [])),
                "n_train_samples": self.disease_meta.get("n_train_samples", 0),
            },
            "version": "v1.0.0",
            "region": "Maharashtra (36 districts)",
        }
