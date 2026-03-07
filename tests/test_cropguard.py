"""
CropGuard ML — Unit Tests
Tests for data loading, features, API, and model predictions.
"""

import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np


# ============================================================
# DATA LOADING TESTS
# ============================================================
class TestDataLoading:

    def test_raw_dataset_exists(self):
        raw_path = PROJECT_ROOT / "data" / "raw" / "maharashtra_agri_survey_2010_2024.csv"
        assert raw_path.exists(), "Raw dataset CSV should exist"

    def test_dataset_shape(self):
        df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "maharashtra_agri_survey_2010_2024.csv")
        assert len(df) > 100, "Dataset should have >100 records"
        assert "district" in df.columns
        assert "yield_kg_per_hectare" in df.columns
        assert "disease_risk_label" in df.columns

    def test_all_districts_present(self):
        from src.constants import ALL_DISTRICTS
        df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "maharashtra_agri_survey_2010_2024.csv")
        generated_districts = set(df["district"].unique())
        assert len(generated_districts) >= 30, f"Only {len(generated_districts)} districts found"

    def test_yield_values_positive(self):
        df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "maharashtra_agri_survey_2010_2024.csv")
        assert (df["yield_kg_per_hectare"] > 0).all(), "All yield values should be positive"

    def test_disease_risk_labels_valid(self):
        df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "maharashtra_agri_survey_2010_2024.csv")
        valid_labels = {"Low", "Medium", "High", "Critical"}
        assert set(df["disease_risk_label"].unique()).issubset(valid_labels)


# ============================================================
# FEATURE ENGINEERING TESTS
# ============================================================
class TestFeatureEngineering:

    def test_rainfall_adequacy(self):
        from src.features import compute_rainfall_adequacy
        df = pd.DataFrame({
            "rainfall_mm_seasonal": [500, 1000],
            "crop_name": ["Soybean", "Rice"]
        })
        result = compute_rainfall_adequacy(df)
        assert len(result) == 2
        assert result.iloc[0] > 0

    def test_vpd_computation(self):
        from src.features import compute_vpd
        df = pd.DataFrame({
            "temp_max_avg_C": [35.0, 25.0],
            "humidity_avg_pct": [50.0, 80.0],
        })
        result = compute_vpd(df)
        assert len(result) == 2
        assert result.iloc[0] > result.iloc[1], "Higher temp + lower humidity = higher VPD"

    def test_irrigation_efficiency(self):
        from src.features import compute_irrigation_efficiency
        df = pd.DataFrame({"irrigation_type": ["drip", "rainfed", "furrow"]})
        result = compute_irrigation_efficiency(df)
        assert result.iloc[0] == 1.0, "Drip should be 1.0"
        assert result.iloc[1] == 0.4, "Rainfed should be 0.4"


# ============================================================
# CONSTANTS TESTS
# ============================================================
class TestConstants:

    def test_36_districts(self):
        from src.constants import ALL_DISTRICTS
        assert len(ALL_DISTRICTS) == 36, f"Expected 36 districts, got {len(ALL_DISTRICTS)}"

    def test_6_divisions(self):
        from src.constants import DIVISIONS
        assert len(DIVISIONS) == 6

    def test_district_division_mapping_complete(self):
        from src.constants import DISTRICT_TO_DIVISION, ALL_DISTRICTS
        for d in ALL_DISTRICTS:
            assert d in DISTRICT_TO_DIVISION, f"Missing mapping for {d}"

    def test_major_crops(self):
        from src.constants import MAJOR_CROPS
        assert len(MAJOR_CROPS) == 8

    def test_crop_diseases(self):
        from src.constants import CROP_DISEASES, MAJOR_CROPS
        for crop in MAJOR_CROPS:
            assert crop in CROP_DISEASES, f"Missing diseases for {crop}"
            assert len(CROP_DISEASES[crop]) >= 2


# ============================================================
# API TESTS
# ============================================================
class TestAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_get_districts(self, client):
        response = client.get("/districts")
        assert response.status_code == 200
        districts = response.json()
        assert len(districts) == 36

    def test_get_crops(self, client):
        response = client.get("/crops/Nashik")
        assert response.status_code == 200
        crops = response.json()
        assert len(crops) > 0

    def test_invalid_district(self, client):
        response = client.get("/crops/InvalidDistrict")
        assert response.status_code == 404

    def test_model_info(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_feedback(self, client):
        response = client.post("/feedback", json={
            "district": "Nashik",
            "crop": "Onion",
            "season": "Rabi",
            "year": 2025,
            "actual_yield_kg_ha": 18000,
            "comments": "Good season",
        })
        assert response.status_code == 200
        assert response.json()["status"] == "received"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
