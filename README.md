# 🌾 CropGuard ML — Maharashtra Agriculture AI Platform

> **Dual-model AI pipeline for crop yield prediction and disease risk assessment across all 36 districts of Maharashtra.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

**CropGuard ML** is an end-to-end machine learning platform designed specifically for **Maharashtra's agriculture ecosystem**. It provides:

| Model | Output | Algorithm |
|-------|--------|-----------|
| **Yield Predictor** | Crop yield in kg/hectare + confidence interval | XGBoost Regressor |
| **Disease Scanner** | Risk score (0–1) + risk label + recommendations | XGBoost Classifier |

### Key Features
- 🗺️ **36 districts** across 6 agro-climatic divisions of Maharashtra
- 🌾 **8+ major crops**: Soybean, Cotton, Rice, Onion, Sugarcane, Grapes, Tur Dal, Jowar
- 🤖 **XGBoost models** with Optuna hyperparameter tuning
- 📊 **SHAP explainability** — top yield/risk drivers per prediction
- 🌐 **FastAPI REST API** with auto-generated Swagger docs
- 📱 **Streamlit dashboard** with farming-themed UI and Marathi (मराठी) labels
- 🐳 **Docker-ready** for containerized deployment
- � **Docker-ready** for containerized deployment
- ⚡ **CI/CD** via GitHub Actions


---

## 🏗️ Project Structure

```
CropGuard ML/
├── api/                    # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── schemas.py          # Pydantic request/response models
│   └── __init__.py
├── app/                    # Streamlit dashboard
│   ├── streamlit_app.py    # 6-page dashboard
│   └── assets/
│       └── style.css       # Farming theme CSS
├── src/                    # Core ML modules
│   ├── constants.py        # Maharashtra districts, crops, soil, climate data
│   ├── data_generator.py   # Synthetic data engine
│   ├── features.py         # Feature engineering pipeline
│   ├── train_yield.py      # Yield regression model training
│   ├── train_disease.py    # Disease classification model training
│   └── predict.py          # Inference / prediction module
├── data/                   # Generated datasets
│   ├── raw/                # Drop real CSVs here (IMD, NBSS, MahaAgri)
│   ├── processed/          # Cleaned merged data
│   ├── synthetic/          # Synthetic training data
│   └── features/           # Train/val/test splits
├── models/                 # Trained model files (.joblib)
├── tests/                  # Unit tests
│   └── test_cropguard.py
├── Dockerfile.api          # API Dockerfile
├── Dockerfile.app          # Dashboard Dockerfile
├── .github/workflows/      # CI/CD
│   └── ci.yml
├── run_pipeline.py         # Master pipeline runner
├── requirements.txt        # Python dependencies
├── .github/                # CI/CD workflows
├── .gitignore
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/cropguard-ml.git
cd "CropGuard ML"
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Data → Train → Ready)

```bash
python run_pipeline.py
```

This generates synthetic data, engineers features, and trains both models (~3-5 minutes).

### 3. Start the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs at **http://localhost:8000/docs**

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict/yield` | Predict crop yield (kg/ha) + confidence interval |
| `POST` | `/predict/disease` | Disease risk score + label + recommendations |
| `GET` | `/districts` | All 36 Maharashtra districts with division mapping |
| `GET` | `/crops/{district}` | Crops grown in a district |
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model version + accuracy metrics |
| `POST` | `/feedback` | Submit actual yield for ground-truth logging |

### Example: Yield Prediction

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

### Example: Disease Risk

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

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Maharashtra yield heatmap, key metrics, season filter |
| 🌾 **Yield Predictor** | Input form → predicted yield + gauge chart |
| 🦠 **Disease Scanner** | Risk score dial + factors + Marathi recommendations |
| 📊 **District Insights** | Historical trends, rainfall vs yield scatter |
| 🗺️ **Division Explorer** | Side-by-side division comparison |
| 📈 **Model Dashboard** | Model accuracy, feature importance, per-division R² |

---

## 🗺️ Maharashtra Coverage

### 6 Agro-Climatic Divisions

| Division | Key Districts | Primary Crops | Soil Type |
|----------|--------------|---------------|-----------|
| **Konkan** | Raigad, Ratnagiri, Sindhudurg | Rice, Coconut, Mango | Laterite, Alluvial |
| **Nashik** | Nashik, Ahmednagar, Dhule | Grapes, Onion, Wheat | Medium Black, Red |
| **Pune** | Pune, Satara, Kolhapur | Sugarcane, Soybean | Deep Black Cotton |
| **Aurangabad** | Chhatrapati Sambhajinagar, Latur, Beed | Soybean, Cotton, Tur Dal | Medium-Deep Black |
| **Amravati** | Amravati, Akola, Yavatmal | Cotton, Soybean, Orange | Deep Black Cotton |
| **Nagpur** | Nagpur, Wardha, Chandrapur | Rice, Cotton, Wheat | Red Laterite, Alluvial |

### 8 Crop-Disease Pairs

| Crop | Diseases Detected |
|------|-------------------|
| Soybean (सोयाबीन) | Yellow Mosaic, Bacterial Pustule, Pod Borer |
| Cotton (कापूस) | Pink Bollworm, Bacterial Blight, Root Rot |
| Rice (भात) | Blast, Brown Planthopper, Sheath Blight |
| Onion (कांदा) | Purple Blotch, Stemphylium, Downy Mildew |
| Sugarcane (ऊस) | Red Rot, Wilt, Smut, Mosaic Virus |
| Grapes (द्राक्षे) | Downy Mildew, Botrytis, Powdery Mildew |
| Tur Dal (तूर) | Fusarium Wilt, Sterility Mosaic |
| Jowar (ज्वारी) | Anthracnose, Downy Mildew, Grain Mold |

---

## 🔬 Feature Engineering

### Derived Features (8 computed features)

| Feature | Formula / Logic |
|---------|----------------|
| `rainfall_adequacy_ratio` | actual_rainfall / crop_water_requirement |
| `growing_degree_days` | Σ((Tmax+Tmin)/2 - T_base) × growing_days |
| `vapour_pressure_deficit` | (1 - RH/100) × 0.6108 × e^(17.27T/(T+237)) |
| `soil_crop_suitability_score` | Rule-based: crop vs soil_type + pH compatibility |
| `irrigation_efficiency_score` | drip=1.0, sprinkler=0.85, furrow=0.65, rainfed=0.4 |
| `pest_pressure_index` | humidity_streak × temp_disease_window / 30 |
| `yield_lag_1yr` | Previous year yield for same (district, crop) |
| `ndvi_anomaly` | ndvi_peak − historical_mean_ndvi_for_district |

### Data Split (Temporal — No Data Leakage)
- **Train**: 2010–2020 (70%)
- **Validation**: 2021–2022 (15%)
- **Test**: 2023–2024 (15%)

---

## 🐳 Docker Deployment

```bash
# Build API
docker build -f Dockerfile.api -t cropguard-api .

# Build Dashboard
docker build -f Dockerfile.app -t cropguard-app .

# Run
docker run -p 8000:8000 cropguard-api
docker run -p 8501:8501 cropguard-app
```

---

## ☁️ Deploy to Any Cloud 

Deploy to AWS, GCP, Azure, or any VPS:
- **API**: Web Service → `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Dashboard**: Web Service → `streamlit run app/streamlit_app.py --server.port=$PORT`

---

## 📦 Data Sources

The project uses real historical data that mirrors Maharashtra's statistical distributions. Data sources compiled in `data/raw/`:

| Source | Data Type | URL | Priority |
|--------|-----------|-----|----------|
| IMD Pune | Rainfall, Temp, Humidity | [imdpune.gov.in](https://imdpune.gov.in) | 🔴 Critical |
| MahaAgri | Crop area, production, yield | [mahaagri.gov.in](https://mahaagri.gov.in) | 🔴 Critical |
| NBSS&LUP | Soil type, pH, organic carbon | [nbsslup.icar.gov.in](https://nbsslup.icar.gov.in) | 🔴 Critical |
| Bhuvan/ISRO | NDVI satellite data | [bhuvan.nrsc.gov.in](https://bhuvan.nrsc.gov.in) | 🟡 High |
| Agmarknet | Mandi prices | [agmarknet.gov.in](https://agmarknet.gov.in) | 🟡 High |

---

## 📊 Model Performance Targets

| Metric | Target | Model |
|--------|--------|-------|
| RMSE (Yield) | < 12% of district mean | Yield Model |
| MAPE (Yield) | < 18% | Yield Model |
| F1-Score | > 0.78 (weighted) | Disease Model |
| Recall (Critical) | > 0.85 | Disease Model |
| Per-division R² | > 0.70 each | Yield Model |
| API Latency | < 200ms | Both |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | XGBoost, scikit-learn |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Class Balancing | SMOTE (imbalanced-learn) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| MLOps | Evidently AI |
| CI/CD | GitHub Actions |
| Deployment | Docker |
| Language | Python 3.11 |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**🌾 CropGuard ML — Built with ❤️ for Maharashtra's Farmers**

**महाराष्ट्रातील शेतकऱ्यांसाठी AI तंत्रज्ञान**

</div>
