# 🌾 CropGuard ML — Maharashtra Agriculture AI Platform

> **Dual-model AI pipeline for crop yield prediction and disease risk assessment across all 36 districts of Maharashtra, India.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?logo=render)](https://cropguard-ml-app.onrender.com)

---

## 🚀 Live Demo

| Service | URL | Status |
|---|---|---|
| 📱 Streamlit Dashboard | [cropguard-ml-app.onrender.com](https://cropguard-ml-app.onrender.com) | ✅ Live |

> ⏳ **Note:** Hosted on Render's free tier — app may take **30–60 seconds** to wake up on first load. Please wait before refreshing.

---

## 📌 Overview

**CropGuard ML** is an end-to-end machine learning platform built specifically for **Maharashtra's agriculture ecosystem**. It solves two critical problems farmers and agriculture officers face every season:

| Model | Problem Solved | Output | Algorithm |
|---|---|---|---|
| 🌾 **Yield Predictor** | *"How much will my crop yield this season?"* | kg/acre + confidence interval + SHAP drivers | XGBoost Regressor |
| 🦠 **Disease Scanner** | *"Is my crop at risk — before symptoms appear?"* | Risk score (0–1) + label + top risk factors | XGBoost Classifier |

### Why Maharashtra-Specific?

Generic India-level models miss critical regional nuance. CropGuard is built around:
- **6 agro-climatic divisions** with distinct soil and climate profiles
- **36 districts** each with their own rainfall long-period averages (LPA)
- **8 major crops** mapped to their actual growing regions
- **Crop-disease pairs** based on Maharashtra's known pest pressure zones (e.g., Pink Bollworm in Yavatmal, Downy Mildew in Nashik vineyards)

---

## 🗺️ Maharashtra Coverage

### 6 Agro-Climatic Divisions × 36 Districts

| Division | Key Districts | Primary Crops | Soil Type |
|---|---|---|---|
| **Konkan** | Raigad, Ratnagiri, Sindhudurg, Thane | Rice, Coconut, Cashew, Mango | Laterite, Alluvial |
| **Nashik** | Nashik, Ahmednagar, Dhule, Nandurbar, Jalgaon | Grapes, Onion, Wheat, Maize | Medium Black, Red |
| **Pune** | Pune, Satara, Sangli, Solapur, Kolhapur | Sugarcane, Soybean, Jowar, Turmeric | Deep Black Cotton |
| **Aurangabad (Marathwada)** | Chhatrapati Sambhajinagar, Latur, Nanded, Osmanabad, Beed | Soybean, Cotton, Tur Dal, Jowar | Medium-Deep Black |
| **Amravati (Vidarbha West)** | Amravati, Akola, Washim, Buldhana, Yavatmal | Cotton, Soybean, Orange, Wheat | Deep Black Cotton |
| **Nagpur (Vidarbha East)** | Nagpur, Wardha, Chandrapur, Gadchiroli, Gondia | Rice, Cotton, Wheat, Teak | Red Laterite, Alluvial |

### 8 Crop–Disease Pairs

| Crop | Marathi | Diseases Detected |
|---|---|---|
| Soybean | सोयाबीन | Yellow Mosaic Virus, Bacterial Pustule, Pod Borer |
| Cotton | कापूस | Pink Bollworm, Bacterial Blight, Root Rot |
| Rice | भात | Blast, Brown Planthopper, Sheath Blight |
| Onion | कांदा | Purple Blotch, Stemphylium Blight, Downy Mildew |
| Sugarcane | ऊस | Red Rot, Wilt, Smut, Mosaic Virus |
| Grapes | द्राक्षे | Downy Mildew, Botrytis Bunch Rot, Powdery Mildew |
| Tur Dal | तूर | Fusarium Wilt, Sterility Mosaic, Phytophthora Blight |
| Jowar | ज्वारी | Anthracnose, Downy Mildew, Grain Mold |

---

## 📊 Data — Honest Overview

### Current Status: Statistically Calibrated Synthetic Data

> **Training data is synthetically generated using statistical distributions derived from published reports by IMD Pune, NBSS&LUP, ICAR, and the Maharashtra Agriculture Department.**
>
> It is **not raw government CSV data** — it is engineered to match the statistical reality of each district: district-specific rainfall LPAs, soil profiles, historical yield ranges, and crop calendars sourced from official publications.

This approach was chosen because:
- Raw district-level IMD data requires formal data-sharing agreements
- NBSS&LUP soil maps are available as GIS layers, not structured CSVs
- Maharashtra Agriculture Department yield data is published in aggregated annual reports, not machine-readable APIs

### Real Data Integration — Roadmap

The ingestion pipeline in `data/raw/` is built and ready. Real data will be integrated in phases:

| Phase | Source | Data Type | Status |
|---|---|---|---|
| Phase 1 | [IMD Pune](https://imdpune.gov.in) | District rainfall + temp + humidity (daily) | 🔄 In Progress — formal request submitted |
| Phase 1 | [MahaAgri Portal](https://mahaagri.gov.in) | Crop area, production, yield (taluka-wise) | 🔄 Downloading open PDFs, parsing in progress |
| Phase 2 | [NBSS&LUP](https://nbsslup.icar.gov.in) | Soil pH, organic carbon, texture (GIS) | 📋 Planned |
| Phase 2 | [Bhuvan / ISRO](https://bhuvan.nrsc.gov.in) | NDVI satellite indices | 📋 Planned — API key requested |
| Phase 3 | [Agmarknet](https://agmarknet.gov.in) | Mandi prices by district + crop | 📋 Planned |
| Phase 3 | [India Water Portal](https://indiawaterportal.org) | Groundwater levels | 📋 Planned |

### How the Synthetic Data Was Built

Not random noise — statistically grounded generation:

```
For each (district, crop, season, year):
  1. Sample rainfall from district-specific LPA distribution (IMD published norms)
  2. Sample soil properties from division-level NBSS published profiles
  3. Apply crop-specific yield response curves (ICAR crop production guidelines)
  4. Inject drought scenarios: Marathwada 2012, 2015, 2018 — rainfall at 60–70% LPA
  5. Inject flood scenarios: Konkan/Kolhapur 2019 — rainfall at 300% LPA in August
  6. Inject disease outbreak scenarios: humidity_streak > 12 days + temp in pathogen window
  7. Add ±8% Gaussian noise to simulate real-world measurement variability
```

This makes the data behaviourally realistic even if not sourced from instruments.

---

## 🏗️ Project Structure

```
CropGuard-ML/
├── api/                        # FastAPI backend
│   ├── main.py                 # All API endpoints
│   ├── schemas.py              # Pydantic request/response models
│   └── __init__.py
├── app/                        # Streamlit dashboard
│   ├── streamlit_app.py        # 6-page dashboard
│   └── assets/                 # CSS and static assets
├── src/                        # Core ML modules
│   ├── constants.py            # 36 districts, crops, soil, climate constants
│   ├── features.py             # Feature engineering pipeline
│   ├── predict.py              # Inference module
│   ├── train_disease.py        # Disease classification model training
│   ├── train_yield.py          # Yield regression model training
│   └── __init__.py
├── data/
│   ├── raw/                    # maharashtra_agr... (real data ingestion target)
│   │   └── feedback_log.json   # Farmer feedback logs
│   ├── processed/
│   │   └── mh_cleaned_merged.csv  # Cleaned + merged district-level data
│   └── features/               # Train / val / test splits
│       ├── train.csv
│       ├── train_engineered.csv
│       ├── val.csv
│       ├── val_engineered.csv
│       ├── test.csv
│       └── test_engineered.csv
├── models/                     # Trained model artifacts
│   ├── yield_model.joblib
│   ├── yield_model_metadata.json
│   ├── disease_model.joblib
│   ├── disease_model_metadata.json
│   └── disease_label_encoder.joblib
├── tests/
│   └── test_cropguard.py       # Unit tests
├── .github/                    # GitHub Actions CI/CD
├── Dockerfile.api
├── Dockerfile.app
├── run_pipeline.py             # Master pipeline: generate → engineer → train
├── requirements.txt
└── README.md
```

---

## ⚙️ Feature Engineering

### 35+ Features Across 5 Categories

**Weather (IMD Pune)**
- `rainfall_mm_seasonal`, `rainfall_deviation_pct` (vs. district LPA)
- `temp_max_avg_C`, `temp_min_avg_C`, `humidity_avg_pct`
- `dry_spell_days`, `onset_monsoon_date` offset

**Soil (NBSS&LUP profiles)**
- `soil_type`, `soil_pH`, `organic_carbon_pct`
- `nitrogen_kg_ha`, `phosphorus_kg_ha`, `potassium_kg_ha`

**Crop Management**
- `irrigation_type`, `seed_variety`, `fertilizer_kg_ha`
- `sowing_date_deviation`, `previous_crop`

**Remote Sensing (Bhuvan NDVI)**
- `ndvi_peak`, `ndvi_sowing`, `ndvi_delta`, `ndvi_stress_weeks`

**Disease-Specific**
- `humidity_streak_days`, `temp_disease_window`
- `prev_season_disease`, `pest_alert_issued`, `pest_pressure_index`

### 8 Derived Features (Computed, Not Collected)

| Feature | Formula | Why It Matters for Maharashtra |
|---|---|---|
| `rainfall_adequacy_ratio` | actual / crop_water_requirement | Marathwada chronically < 1.0 |
| `growing_degree_days` | Σ((Tmax+Tmin)/2 − T_base) | Grape maturity in Nashik, sugarcane in Kolhapur |
| `vapour_pressure_deficit` | (1−RH/100) × 0.6108 × e^(17.27T/(T+237)) | Key trigger for Downy Mildew on grapes and onion |
| `soil_crop_suitability_score` | Rule-based: crop vs. soil compatibility | Cotton on Black Cotton soil = high; penalizes mismatches |
| `irrigation_efficiency_score` | drip=1.0 / sprinkler=0.85 / furrow=0.65 / rainfed=0.4 | Reflects MH government drip subsidy adoption |
| `pest_pressure_index` | humidity_streak × temp_disease_window / 30 | Combined disease trigger score |
| `yield_lag_1yr` | Previous year yield (district, crop) | Captures soil depletion and farmer practice trends |
| `ndvi_anomaly` | ndvi_peak − district_historical_mean | Early stress signal before visible crop damage |

### Data Split — Temporal, Zero Leakage

```
Train:      2010–2020  (70%)   ← model learns historical patterns
Validation: 2021–2022  (15%)   ← hyperparameter tuning
Test:       2023–2024  (15%)   ← final evaluation, never touched during training
```

> Split is **by year, not random** — prevents data leakage from temporal autocorrelation.

---

## 🤖 Model Architecture

### Model 1 — Yield Prediction (XGBoost Regressor)

```python
Target:    yield_kg_per_acre
Baseline:  district historical average yield (naive baseline to beat)
Tuning:    Optuna (300 trials, TimeSeriesSplit k=5)
Metrics:   RMSE, MAPE, R² per division
```

**Hyperparameters tuned:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
`colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`

### Model 2 — Disease Risk (XGBoost Classifier)

```python
Target:    disease_risk_label → Low / Medium / High / Critical
Imbalance: SMOTE + class_weight (Critical cases are rare)
Threshold: Tuned per crop to achieve recall ≥ 0.85 on Critical class
Output:    risk_score (0.0–1.0) + risk_label + top_3_risk_factors
```

> Separate classifier trained per major crop — soybean, cotton, rice, onion — because pathogen conditions differ significantly across crops.

### Explainability — SHAP

Every prediction returns the top 3 SHAP-derived factors driving the output. No black box.

```json
{
  "predicted_yield_kg_acre": 18450,
  "top_yield_drivers": [
    "rainfall_adequacy_ratio (+2340 kg/acre)",
    "ndvi_peak (+1180 kg/acre)",
    "irrigation_type: drip (+890 kg/acre)"
  ]
}
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/shindelucky40-cmyk/CropGuard-ML.git
cd CropGuard-ML
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python run_pipeline.py
```

Generates synthetic data → engineers features → trains both models → saves to `models/`. Takes ~3–5 minutes.

### 3. Start Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**

### 4. Start API

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger docs at **http://localhost:8000/docs**

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict/yield` | Yield prediction + confidence interval + SHAP drivers |
| `POST` | `/predict/disease` | Disease risk score + label + recommendations |
| `GET` | `/districts` | All 36 MH districts with division mapping |
| `GET` | `/crops/{district}` | Crops grown in a specific district |
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model version, training date, accuracy metrics |
| `POST` | `/feedback` | Submit actual yield for ground-truth logging |

### Example — Yield Prediction (Nashik Onion)

```bash
curl -X POST http://localhost:8000/predict/yield \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Nashik",
    "crop": "Onion",
    "season": "Rabi",
    "year": 2025,
    "rainfall_mm_seasonal": 420.5,
    "soil_pH": 7.1,
    "irrigation_type": "drip",
    "fertilizer_kg_ha": 180,
    "sowing_date_deviation": -3,
    "ndvi_peak": 0.72
  }'
```

### Example — Disease Risk (Yavatmal Cotton)

```bash
curl -X POST http://localhost:8000/predict/disease \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Yavatmal",
    "crop": "Cotton",
    "humidity_avg_pct": 82,
    "temp_max_avg_C": 33.5,
    "humidity_streak_days": 12,
    "temp_disease_window": 15,
    "ndvi_peak": 0.55,
    "ndvi_stress_weeks": 3,
    "prev_season_disease": 1,
    "pest_alert_issued": 0
  }'
```

---

## 📱 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Maharashtra yield heatmap by district, season selector, division filter |
| 🌾 **Yield Predictor** | Input form → predicted yield + gauge chart + SHAP drivers |
| 🦠 **Disease Scanner** | Risk score dial + top risk factors + Marathi recommendations |
| 📊 **District Insights** | Historical yield trends (2010–2024), rainfall vs. yield scatter |
| 🗺️ **Division Explorer** | Side-by-side comparison across all 6 divisions |
| 📈 **Model Dashboard** | Accuracy metrics, feature importance chart, per-division R², drift status |

---

## 📈 Model Performance

> Metrics are on the held-out test set (2023–2024 data — never seen during training).

| Metric | Target | Model |
|---|---|---|
| RMSE | < 12% of district mean yield | Yield Model |
| MAPE | < 18% | Yield Model |
| F1-Score (weighted) | > 0.78 | Disease Model |
| Recall (Critical class) | > 0.85 | Disease Model |
| Per-division R² | > 0.70 for each division | Yield Model |
| API latency | < 200ms | Both |

---

## 🐳 Docker

```bash
# Build and run API
docker build -f Dockerfile.api -t cropguard-api .
docker run -p 8000:8000 cropguard-api

# Build and run Dashboard
docker build -f Dockerfile.app -t cropguard-app .
docker run -p 8501:8501 cropguard-app
```

---

## ⚙️ MLOps

| Component | Tool | Purpose |
|---|---|---|
| CI/CD | GitHub Actions | Lint → Test → Train → Evaluate → Deploy on push to main |
| Containerization | Docker | Reproducible API + Dashboard environments |
| Hosting | Render | Free-tier cloud deployment |

### CI/CD Flow

```
Push to main
  → Lint + unit tests (pytest)
  → Data validation (schema check)
  → Model training
  → Docker build → push → Render deploy
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, Scikit-learn, Optuna, SHAP, imbalanced-learn (SMOTE) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| CI/CD | GitHub Actions |
| Containers | Docker |
| Deployment | Render |
| Language | Python 3.11 |

---



## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'feat: add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---


## 👤 Author

**Lalit shinde** — B.Tech AI/ML | Aspiring ML Engineer.  


---

> 🌾 **CropGuard ML — Built for Maharashtra's Farmers**
>
> महाराष्ट्रातील शेतकऱ्यांसाठी AI तंत्रज्ञान 🇮🇳>

