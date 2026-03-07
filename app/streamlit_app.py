"""
CropGuard ML — Streamlit Dashboard
Maharashtra Agriculture AI Platform
"""

import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

from src.constants import (
    DIVISIONS, DISTRICT_TO_DIVISION, ALL_DISTRICTS, MAJOR_CROPS,
    DIVISION_CROPS, CROP_SEASON, CROP_MARATHI, CROP_DISEASES,
    SEASONS, CROP_AVG_YIELD, RISK_LABELS_MARATHI, DIVISION_RAINFALL_LPA,
    IRRIGATION_TYPES, CROP_WATER_REQUIREMENT, IRRIGATION_EFFICIENCY
)
from src.predict import CropGuardPredictor

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CropGuard ML — Maharashtra Agriculture AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ============================================================
# LANGUAGE SYSTEM
# ============================================================
LANG = {
    "English": {
        "app_title": "CropGuard ML",
        "app_subtitle": "AI-Powered Crop Intelligence for Maharashtra",
        "home": "Home", "yield_pred": "Yield Predictor", "disease_scan": "Disease Scanner",
        "district_ins": "District Insights", "div_explore": "Division Explorer", "model_dash": "Model Dashboard",
        "districts_covered": "Districts Covered", "major_crops": "Major Crops",
        "ai_models": "AI Models", "divisions": "Divisions",
        "select_crop": "Select Crop", "select_season": "Select Season",
        "select_district": "Select District", "enter_details": "Enter Details",
        "predict_yield": "Predict Yield", "prediction_result": "Prediction Result",
        "scan_disease": "Scan Disease Risk", "risk_assessment": "Risk Assessment",
        "weather": "Weather Conditions", "crop_mgmt": "Crop Management",
        "satellite": "Satellite Data", "scan_params": "Scan Parameters",
        "current_cond": "Current Conditions", "field_status": "Field Status",
        "risk_history": "Risk History", "recommendations": "Recommendations",
        "top_drivers": "Top Yield Drivers", "probable_diseases": "Probable Diseases",
        "top_risk_factors": "Top Risk Factors", "summary_stats": "Summary Statistics",
        "built_for": "Built for Indian Farmers",
        "tip": "For best accuracy, use the latest weather data from IMD Pune and actual soil test results from your local KVK.",
    },
    "Marathi": {
        "app_title": "CropGuard ML",
        "app_subtitle": "महाराष्ट्रासाठी AI-आधारित पीक बुद्धिमत्ता",
        "home": "मुख्यपृष्ठ", "yield_pred": "उत्पादन अंदाज", "disease_scan": "रोग स्कॅनर",
        "district_ins": "जिल्हा माहिती", "div_explore": "विभाग तुलना", "model_dash": "मॉडेल माहिती",
        "districts_covered": "जिल्हे", "major_crops": "प्रमुख पिके",
        "ai_models": "AI मॉडेल", "divisions": "विभाग",
        "select_crop": "पीक निवडा", "select_season": "हंगाम निवडा",
        "select_district": "जिल्हा निवडा", "enter_details": "तपशील भरा",
        "predict_yield": "उत्पादन अंदाज करा", "prediction_result": "अंदाज निकाल",
        "scan_disease": "रोग तपासा", "risk_assessment": "जोखीम मूल्यांकन",
        "weather": "हवामान स्थिती", "crop_mgmt": "पीक व्यवस्थापन",
        "satellite": "उपग्रह डेटा", "scan_params": "स्कॅन पॅरामीटर्स",
        "current_cond": "सध्याची स्थिती", "field_status": "शेत स्थिती",
        "risk_history": "जोखीम इतिहास", "recommendations": "शिफारशी",
        "top_drivers": "प्रमुख उत्पादन घटक", "probable_diseases": "संभाव्य रोग",
        "top_risk_factors": "प्रमुख जोखीम घटक", "summary_stats": "सारांश आकडेवारी",
        "built_for": "भारतीय शेतकऱ्यांसाठी",
        "tip": "सर्वोत्तम अचूकतेसाठी, IMD पुणे येथून ताजे हवामान डेटा आणि KVK कडून माती चाचणी वापरा.",
    },
    "Hindi": {
        "app_title": "CropGuard ML",
        "app_subtitle": "महाराष्ट्र के लिए AI-संचालित फसल बुद्धिमत्ता",
        "home": "होम", "yield_pred": "उपज भविष्यवाणी", "disease_scan": "रोग स्कैनर",
        "district_ins": "जिला जानकारी", "div_explore": "विभाग तुलना", "model_dash": "मॉडल जानकारी",
        "districts_covered": "जिले", "major_crops": "प्रमुख फसलें",
        "ai_models": "AI मॉडल", "divisions": "विभाग",
        "select_crop": "फसल चुनें", "select_season": "मौसम चुनें",
        "select_district": "जिला चुनें", "enter_details": "विवरण भरें",
        "predict_yield": "उपज का अनुमान लगाएं", "prediction_result": "भविष्यवाणी परिणाम",
        "scan_disease": "रोग स्कैन करें", "risk_assessment": "जोखिम मूल्यांकन",
        "weather": "मौसम की स्थिति", "crop_mgmt": "फसल प्रबंधन",
        "satellite": "उपग्रह डेटा", "scan_params": "स्कैन पैरामीटर",
        "current_cond": "वर्तमान स्थिति", "field_status": "खेत की स्थिति",
        "risk_history": "जोखिम इतिहास", "recommendations": "सिफारिशें",
        "top_drivers": "प्रमुख उपज कारक", "probable_diseases": "संभावित रोग",
        "top_risk_factors": "प्रमुख जोखिम कारक", "summary_stats": "सारांश आंकड़े",
        "built_for": "भारतीय किसानों के लिए",
        "tip": "सर्वोत्तम सटीकता के लिए, IMD पुणे से ताजा मौसम डेटा और KVK से मिट्टी परीक्षण का उपयोग करें.",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "English"

def t(key):
    """Get translated string."""
    return LANG.get(st.session_state.lang, LANG["English"]).get(key, key)

# Load predictor
@st.cache_resource
def load_predictor():
    model_dir = os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models"))
    return CropGuardPredictor(model_dir=model_dir)

predictor = load_predictor()

# Load data
@st.cache_data
def load_data():
    data_path = PROJECT_ROOT / "data" / "processed" / "mh_cleaned_merged.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

df = load_data()


# ============================================================
# LANGUAGE TOGGLE (top-right)
# ============================================================
lang_col1, lang_col2 = st.columns([5, 1])
with lang_col2:
    selected_lang = st.selectbox(
        "🌐", ["English", "Marathi", "Hindi"],
        index=["English", "Marathi", "Hindi"].index(st.session_state.lang),
        key="lang_selector", label_visibility="collapsed",
    )
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

# ============================================================
# SIDEBAR — Styled Navigation Buttons
# ============================================================
NAV_ITEMS = [
    ("home", "🏠", "home"),
    ("yield_pred", "🌾", "yield"),
    ("disease_scan", "🦠", "disease"),
    ("district_ins", "📊", "district"),
    ("div_explore", "🗺️", "division"),
    ("model_dash", "📈", "model"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "home"

# Inject sidebar button CSS
st.markdown("""
<style>
.nav-btn {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 18px; margin: 4px 0;
    border-radius: 12px; cursor: pointer;
    font-family: 'Outfit', sans-serif; font-size: 1rem; font-weight: 500;
    color: #E8F5E9; background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.25s ease; text-decoration: none;
    width: 100%;
}
.nav-btn:hover {
    background: rgba(255,255,255,0.18);
    transform: translateX(4px);
    border-color: rgba(255,255,255,0.25);
}
.nav-btn.active {
    background: linear-gradient(135deg, #4CAF50, #66BB6A) !important;
    color: white !important; font-weight: 600;
    border-color: #81C784;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
}
.nav-btn .nav-icon { font-size: 1.2rem; }
.nav-btn .nav-label { flex: 1; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌾 CropGuard ML")
    st.markdown(f"**{t('app_subtitle')}**")
    st.markdown("---")

    for label_key, icon, page_id in NAV_ITEMS:
        active_cls = "active" if st.session_state.active_page == page_id else ""
        if st.button(f"{icon}  {t(label_key)}", key=f"nav_{page_id}", use_container_width=True):
            st.session_state.active_page = page_id
            st.rerun()

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; opacity:0.7; font-size:0.8rem;'>"
        f"CropGuard ML v1.0<br>Maharashtra Region<br>"
        f"🇮🇳 {t('built_for')}"
        f"</div>",
        unsafe_allow_html=True,
    )

page = st.session_state.active_page


# ============================================================
# PAGE: HOME
# ============================================================
def page_home():
    st.markdown(
        """<div class="main-header">
            <h1>🌾 CropGuard ML</h1>
            <p>AI-Powered Crop Intelligence for Maharashtra | महाराष्ट्र कृषी AI</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">36</div>'
            '<div class="metric-label">Districts Covered</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">8+</div>'
            '<div class="metric-label">Major Crops</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">2</div>'
            '<div class="metric-label">AI Models</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">6</div>'
            '<div class="metric-label">Divisions</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Maharashtra Map — Yield Heatmap
    st.markdown(
        '<div class="section-header"><h3>📍 Maharashtra Yield Overview</h3></div>',
        unsafe_allow_html=True,
    )

    if df is not None:
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            selected_crop = st.selectbox("Select Crop", MAJOR_CROPS, key="home_crop")
        with col_filter2:
            selected_season = st.selectbox("Select Season",
                                           ["All"] + list(SEASONS.keys()),
                                           key="home_season")

        # Filter and aggregate
        map_df = df.copy()
        map_df = map_df[map_df["crop_name"] == selected_crop]
        if selected_season != "All":
            map_df = map_df[map_df["season"] == selected_season]

        if len(map_df) > 0:
            district_yield = map_df.groupby("district")["yield_kg_per_hectare"].mean().reset_index()
            district_yield.columns = ["District", "Avg Yield (kg/ha)"]
            district_yield = district_yield.sort_values("Avg Yield (kg/ha)", ascending=False)

            # Bar chart as map substitute (GeoJSON not available without geopandas)
            fig = px.bar(
                district_yield,
                x="District",
                y="Avg Yield (kg/ha)",
                color="Avg Yield (kg/ha)",
                color_continuous_scale=["#D32F2F", "#FF9800", "#FFC107", "#8BC34A", "#2E7D32"],
                title=f"Average {selected_crop} Yield by District",
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=450,
                font=dict(family="Outfit"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Division summary table
            div_yield = map_df.groupby("division").agg(
                avg_yield=("yield_kg_per_hectare", "mean"),
                min_yield=("yield_kg_per_hectare", "min"),
                max_yield=("yield_kg_per_hectare", "max"),
                records=("yield_kg_per_hectare", "count"),
            ).round(1).reset_index()
            div_yield.columns = ["Division", "Avg Yield", "Min Yield", "Max Yield", "Records"]
            st.dataframe(div_yield, use_container_width=True, hide_index=True)
        else:
            st.info(f"No data available for {selected_crop} in {selected_season} season.")
    else:
        st.warning("Dataset not found. Run `python run_pipeline.py` first.")

    # How it works section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header"><h3>🔬 How CropGuard Works</h3></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="division-card"><h4>📊 Data Collection</h4>'
            '<p>Weather, soil, NDVI satellite data from IMD, NBSS & Bhuvan for all 36 districts</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="division-card"><h4>🤖 AI Prediction</h4>'
            '<p>XGBoost models trained on 15 years of Maharashtra agricultural data</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="division-card"><h4>🌾 Actionable Insights</h4>'
            '<p>Yield forecasts + disease early warnings with Marathi recommendations</p></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: YIELD PREDICTOR
# ============================================================
def page_yield_predictor():
    st.markdown(
        """<div class="main-header">
            <h1>🌾 Yield Predictor</h1>
            <p>उत्पादन अंदाज | Predict crop yield for your district</p>
        </div>""",
        unsafe_allow_html=True,
    )

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("### 📝 Enter Details")

        district = st.selectbox("District (जिल्हा)", ALL_DISTRICTS, key="yp_district")
        division = DISTRICT_TO_DIVISION.get(district, "")

        available_crops = DIVISION_CROPS.get(division, MAJOR_CROPS[:4])
        all_crops = list(set(available_crops + MAJOR_CROPS))
        crop = st.selectbox("Crop (पीक)", all_crops, key="yp_crop")
        if crop in CROP_MARATHI:
            st.caption(f"मराठी: {CROP_MARATHI[crop]}")

        season = st.selectbox("Season (हंगाम)", list(SEASONS.keys()), key="yp_season")

        st.markdown("#### 🌧️ Weather Conditions")
        rainfall = st.slider("Seasonal Rainfall (mm)", 100, 3000,
                              int(DIVISION_RAINFALL_LPA.get(division, 700) * 0.75),
                              key="yp_rain")
        temp_max = st.slider("Avg Max Temperature (°C)", 20, 45, 33, key="yp_tmax")
        temp_min = st.slider("Avg Min Temperature (°C)", 5, 30, 20, key="yp_tmin")
        humidity = st.slider("Avg Humidity (%)", 20, 95, 60, key="yp_hum")

        st.markdown("#### 🌱 Crop Management")
        irrigation = st.selectbox("Irrigation Type", IRRIGATION_TYPES, key="yp_irr")
        fertilizer = st.number_input("Fertilizer (kg/ha)", 0, 500, 150, key="yp_fert")
        sowing_dev = st.slider("Sowing Date Deviation (days)", -15, 15, 0, key="yp_sow")

        st.markdown("#### 🛰️ Satellite Data")
        ndvi_peak = st.slider("NDVI Peak", 0.1, 0.95, 0.65, 0.01, key="yp_ndvi")
        soil_ph = st.slider("Soil pH", 4.0, 9.5, 7.0, 0.1, key="yp_ph")

        predict_clicked = st.button("🌾 Predict Yield", use_container_width=True, key="yp_btn")

    with col_result:
        st.markdown("### 📊 Prediction Result")

        if predict_clicked:
            input_data = {
                "district": district, "crop": crop, "season": season,
                "division": division, "year": 2025,
                "rainfall_mm_seasonal": rainfall,
                "temp_max_avg_C": temp_max, "temp_min_avg_C": temp_min,
                "humidity_avg_pct": humidity,
                "irrigation_type": irrigation,
                "fertilizer_kg_ha": fertilizer,
                "sowing_date_deviation": sowing_dev,
                "ndvi_peak": ndvi_peak, "soil_pH": soil_ph,
                "dry_spell_days": 5,
                "rainfall_deviation_pct": 0,
                "onset_monsoon_deviation_days": 0,
                "fog_days": 0,
                "organic_carbon_pct": 0.5,
                "nitrogen_kg_ha": 200,
                "phosphorus_kg_ha": 20,
                "potassium_kg_ha": 300,
                "soil_moisture_pct": 25,
                "ec_dS_m": 0.5,
                "seed_variety": "HYV",
                "pesticide_applications": 2,
                "previous_crop": "Wheat",
                "ndvi_sowing": 0.3,
                "ndvi_delta": ndvi_peak - 0.3,
                "ndvi_stress_weeks": 1,
                "humidity_streak_days": 5,
                "temp_disease_window": 10,
                "prev_season_disease": 0,
                "pest_alert_issued": 0,
                "rainfall_adequacy_ratio": rainfall / max(1, CROP_WATER_REQUIREMENT.get(crop, 500)),
                "growing_degree_days": max(0, ((temp_max + temp_min) / 2 - 10) * 120),
                "vapour_pressure_deficit": (1 - humidity / 100) * 0.6108 * np.exp(17.27 * temp_max / (temp_max + 237.3)),
                "soil_crop_suitability_score": 0.7,
                "irrigation_efficiency_score": IRRIGATION_EFFICIENCY.get(irrigation, 0.5),
                "pest_pressure_index": 2.0,
                "yield_lag_1yr": CROP_AVG_YIELD.get(crop, 1500),
                "ndvi_anomaly": 0,
            }

            result = predictor.predict_yield(input_data)

            if "error" not in result:
                pred_ha = result["predicted_yield_kg_ha"]
                ci_ha = result["confidence_interval"]
                avg_ha = result["district_avg_yield"]

                # Convert to kg/acre (1 hectare = 2.471 acres)
                HA_TO_ACRE = 2.471
                pred = round(pred_ha / HA_TO_ACRE, 1)
                ci = [round(ci_ha[0] / HA_TO_ACRE, 1), round(ci_ha[1] / HA_TO_ACRE, 1)]
                avg = round(avg_ha / HA_TO_ACRE, 1)

                # Big prediction display
                st.markdown(
                    f'<div class="prediction-card yield-result">'
                    f'<div class="yield-value">{pred:,.0f} <span class="yield-unit">kg/acre</span></div>'
                    f'<p style="text-align:center; color:#666;">Confidence: {ci[0]:,.0f} — {ci[1]:,.0f} kg/acre</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred,
                    delta={"reference": avg, "relative": True, "valueformat": ".1%"},
                    title={"text": f"{crop} Yield ({district})"},
                    gauge={
                        "axis": {"range": [0, avg * 2]},
                        "bar": {"color": "#2E7D32"},
                        "steps": [
                            {"range": [0, avg * 0.7], "color": "#FFCDD2"},
                            {"range": [avg * 0.7, avg * 1.3], "color": "#C8E6C9"},
                            {"range": [avg * 1.3, avg * 2], "color": "#81C784"},
                        ],
                        "threshold": {
                            "line": {"color": "#BF360C", "width": 3},
                            "thickness": 0.8,
                            "value": avg,
                        },
                    },
                ))
                fig.update_layout(height=300, font=dict(family="Outfit"))
                st.plotly_chart(fig, use_container_width=True)

                # Top drivers
                st.markdown("**🔑 Top Yield Drivers:**")
                for i, driver in enumerate(result.get("top_yield_drivers", []), 1):
                    st.markdown(f"  {i}. `{driver}`")

                # Comparison with average
                diff = ((pred - avg) / avg) * 100
                if diff > 0:
                    st.success(f"📈 **{diff:.1f}% above** district average ({avg:,.0f} kg/acre)")
                else:
                    st.warning(f"📉 **{abs(diff):.1f}% below** district average ({avg:,.0f} kg/acre)")
            else:
                st.error(f"Model error: {result['error']}")
        else:
            st.info("👈 Fill in the details and click **Predict Yield** to get your forecast.")
            st.markdown(
                '<div class="recommendation-box">'
                '💡 <strong>Tip:</strong> For best accuracy, use the latest weather data '
                'from IMD Pune and actual soil test results from your local KVK.'
                '<br><span class="marathi-text">सर्वोत्तम अचूकतेसाठी, IMD पुणे येथून ताजे हवामान डेटा वापरा</span>'
                '</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE: DISEASE SCANNER
# ============================================================
def page_disease_scanner():
    st.markdown(
        """<div class="main-header">
            <h1>🦠 Disease Risk Scanner</h1>
            <p>रोग जोखीम स्कॅनर | Early disease detection for Maharashtra crops</p>
        </div>""",
        unsafe_allow_html=True,
    )

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("### 🔍 Scan Parameters")

        district = st.selectbox("District (जिल्हा)", ALL_DISTRICTS, key="ds_district")
        division = DISTRICT_TO_DIVISION.get(district, "")

        available_crops = DIVISION_CROPS.get(division, MAJOR_CROPS[:4])
        all_crops = list(set(available_crops + MAJOR_CROPS))
        crop = st.selectbox("Crop (पीक)", all_crops, key="ds_crop")
        if crop in CROP_MARATHI:
            st.caption(f"मराठी: {CROP_MARATHI[crop]}")

        st.markdown("#### 🌡️ Current Conditions")
        humidity = st.slider("Current Humidity (%)", 20, 100, 70, key="ds_hum")
        temp_max = st.slider("Max Temperature (°C)", 15, 48, 32, key="ds_temp")
        humidity_streak = st.slider("High Humidity Streak (days >80%)", 0, 30, 5, key="ds_streak")
        temp_window = st.slider("Days in Disease Temp Window", 0, 30, 10, key="ds_window")

        st.markdown("#### 🛰️ Field Status")
        ndvi_peak = st.slider("Current NDVI", 0.1, 0.95, 0.6, 0.01, key="ds_ndvi")
        ndvi_stress = st.slider("NDVI Stress Weeks", 0, 8, 1, key="ds_stress")

        st.markdown("#### ⚠️ Risk History")
        prev_disease = st.selectbox("Disease last season?", [0, 1],
                                     format_func=lambda x: "Yes" if x else "No",
                                     key="ds_prev")
        pest_alert = st.selectbox("Govt pest alert issued?", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No",
                                   key="ds_alert")

        scan_clicked = st.button("🦠 Scan Disease Risk", use_container_width=True, key="ds_btn")

    with col_result:
        st.markdown("### 🏥 Risk Assessment")

        if scan_clicked:
            input_data = {
                "district": district, "crop": crop, "division": division,
                "humidity_avg_pct": humidity, "temp_max_avg_C": temp_max,
                "humidity_streak_days": humidity_streak,
                "temp_disease_window": temp_window,
                "ndvi_peak": ndvi_peak, "ndvi_stress_weeks": ndvi_stress,
                "prev_season_disease": prev_disease,
                "pest_alert_issued": pest_alert,
                "rainfall_mm_seasonal": 600,
                "soil_moisture_pct": 25,
                "temp_min_avg_C": 20,
                "dry_spell_days": 5,
                "rainfall_deviation_pct": 0,
                "onset_monsoon_deviation_days": 0,
                "fog_days": 0,
                "soil_pH": 7.0,
                "organic_carbon_pct": 0.5,
                "nitrogen_kg_ha": 200,
                "phosphorus_kg_ha": 20,
                "potassium_kg_ha": 300,
                "ec_dS_m": 0.5,
                "sowing_date_deviation": 0,
                "fertilizer_kg_ha": 150,
                "pesticide_applications": 2,
                "ndvi_sowing": 0.3,
                "ndvi_delta": ndvi_peak - 0.3,
                "rainfall_adequacy_ratio": 1.0,
                "growing_degree_days": 2000,
                "vapour_pressure_deficit": 1.5,
                "soil_crop_suitability_score": 0.7,
                "irrigation_efficiency_score": 0.5,
                "pest_pressure_index": humidity_streak * temp_window / 30,
                "yield_lag_1yr": CROP_AVG_YIELD.get(crop, 1500),
                "ndvi_anomaly": 0,
            }

            result = predictor.predict_disease(input_data)

            if "error" not in result:
                risk_score = result["risk_score"]
                risk_label = result["risk_label"]
                risk_class = risk_label.lower()

                # Risk badge
                st.markdown(
                    f'<div class="prediction-card disease-result-{risk_class}">'
                    f'<div style="text-align:center;">'
                    f'<span class="risk-badge risk-{risk_class}">{risk_label}</span>'
                    f'<p class="marathi-text">{result.get("risk_label_marathi", "")}</p>'
                    f'</div>'
                    f'<div class="yield-value" style="font-size:2.5rem;">{risk_score:.1%}</div>'
                    f'<p style="text-align:center;color:#666;">Disease Risk Score</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    number={"suffix": "%"},
                    title={"text": f"Disease Risk — {crop} ({district})"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#333"},
                        "steps": [
                            {"range": [0, 25], "color": "#4CAF50"},
                            {"range": [25, 50], "color": "#FFC107"},
                            {"range": [50, 75], "color": "#FF9800"},
                            {"range": [75, 100], "color": "#D32F2F"},
                        ],
                    },
                ))
                fig.update_layout(height=280, font=dict(family="Outfit"))
                st.plotly_chart(fig, use_container_width=True)

                # Probable diseases
                diseases = result.get("probable_diseases", [])
                if diseases:
                    st.markdown("**🦠 Probable Diseases:**")
                    for d in diseases:
                        st.markdown(f"  - {d}")

                # Top risk factors
                factors = result.get("top_risk_factors", [])
                if factors:
                    st.markdown("**⚠️ Top Risk Factors:**")
                    for f in factors:
                        st.markdown(f"  - `{f}`")

                # Recommendations
                recs = result.get("recommendations", [])
                if recs:
                    box_class = "critical" if risk_class == "critical" else ("warning" if risk_class == "high" else "")
                    st.markdown(f"**💡 Recommendations (शिफारशी):**")
                    for r in recs:
                        st.markdown(
                            f'<div class="recommendation-box {box_class}">{r}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.error(f"Model error: {result['error']}")
        else:
            st.info("👈 Enter current field conditions and click **Scan Disease Risk**")
            # Show common diseases for reference
            st.markdown("**📋 Common Diseases by Crop:**")
            for crop_name, diseases in list(CROP_DISEASES.items())[:6]:
                marathi = CROP_MARATHI.get(crop_name, "")
                st.markdown(f"**{crop_name}** ({marathi}): {', '.join(diseases)}")


# ============================================================
# PAGE: DISTRICT INSIGHTS
# ============================================================
def page_district_insights():
    st.markdown(
        """<div class="main-header">
            <h1>📊 District Insights</h1>
            <p>जिल्हा माहिती | Historical yield trends and analysis</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Dataset not found. Run `python run_pipeline.py` first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        district = st.selectbox("Select District", ALL_DISTRICTS, key="di_district")
    with col2:
        crop = st.selectbox("Select Crop", MAJOR_CROPS, key="di_crop")

    dist_df = df[(df["district"] == district) & (df["crop_name"] == crop)]

    if len(dist_df) == 0:
        st.info(f"No data for {crop} in {district}. Try another combination.")
        return

    # Yield trend
    yearly = dist_df.groupby("year")["yield_kg_per_hectare"].mean().reset_index()
    fig_trend = px.line(
        yearly, x="year", y="yield_kg_per_hectare",
        title=f"{crop} Yield Trend in {district}",
        markers=True,
        color_discrete_sequence=["#2E7D32"],
    )
    fig_trend.update_layout(
        xaxis_title="Year", yaxis_title="Yield (kg/ha)",
        font=dict(family="Outfit"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Rainfall vs Yield scatter
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        fig_scatter = px.scatter(
            dist_df, x="rainfall_mm_seasonal", y="yield_kg_per_hectare",
            color="season", size="ndvi_peak",
            title="Rainfall vs Yield",
            color_discrete_sequence=["#2E7D32", "#FF9800", "#2196F3"],
        )
        fig_scatter.update_layout(font=dict(family="Outfit"), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_sc2:
        fig_disease = px.histogram(
            dist_df, x="disease_risk_label",
            color="disease_risk_label",
            title="Disease Risk Distribution",
            color_discrete_map={
                "Low": "#4CAF50", "Medium": "#FFC107",
                "High": "#FF9800", "Critical": "#D32F2F"
            },
            category_orders={"disease_risk_label": ["Low", "Medium", "High", "Critical"]},
        )
        fig_disease.update_layout(font=dict(family="Outfit"), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_disease, use_container_width=True)

    # Summary stats
    st.markdown("### 📋 Summary Statistics")
    summary = dist_df[["yield_kg_per_hectare", "rainfall_mm_seasonal",
                         "ndvi_peak", "disease_risk_score"]].describe().round(2)
    st.dataframe(summary, use_container_width=True)


# ============================================================
# PAGE: DIVISION EXPLORER
# ============================================================
def page_division_explorer():
    st.markdown(
        """<div class="main-header">
            <h1>🗺️ Division Explorer</h1>
            <p>विभाग तुलना | Compare Maharashtra's 6 agro-climatic divisions</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("⚠️ Dataset not found.")
        return

    # Division cards
    cols = st.columns(3)
    for i, (div, districts) in enumerate(DIVISIONS.items()):
        with cols[i % 3]:
            div_data = df[df["division"] == div]
            avg_yield = div_data["yield_kg_per_hectare"].mean() if len(div_data) > 0 else 0
            n_districts = len(districts)
            crops = ", ".join(DIVISION_CROPS.get(div, [])[:3])

            st.markdown(
                f'<div class="division-card">'
                f'<h4>{div}</h4>'
                f'<p><strong>{n_districts}</strong> districts</p>'
                f'<p>Avg Yield: <strong>{avg_yield:,.0f}</strong> kg/ha</p>'
                f'<p style="font-size:0.8rem;color:#666;">{crops}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison chart
    crop_filter = st.selectbox("Compare by Crop", ["All"] + MAJOR_CROPS, key="de_crop")
    comp_df = df.copy()
    if crop_filter != "All":
        comp_df = comp_df[comp_df["crop_name"] == crop_filter]

    if len(comp_df) > 0:
        div_comp = comp_df.groupby("division").agg(
            avg_yield=("yield_kg_per_hectare", "mean"),
            avg_rainfall=("rainfall_mm_seasonal", "mean"),
            avg_disease_risk=("disease_risk_score", "mean"),
        ).round(1).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig_yield = px.bar(
                div_comp, x="division", y="avg_yield",
                color="avg_yield", title="Average Yield by Division",
                color_continuous_scale=["#FFCDD2", "#4CAF50", "#1B5E20"],
            )
            fig_yield.update_layout(font=dict(family="Outfit"), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_yield, use_container_width=True)

        with col2:
            fig_risk = px.bar(
                div_comp, x="division", y="avg_disease_risk",
                color="avg_disease_risk", title="Average Disease Risk by Division",
                color_continuous_scale=["#4CAF50", "#FFC107", "#D32F2F"],
            )
            fig_risk.update_layout(font=dict(family="Outfit"), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_risk, use_container_width=True)

        # Rainfall comparison
        fig_rain = px.bar(
            div_comp, x="division", y="avg_rainfall",
            color="division", title="Average Seasonal Rainfall by Division",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_rain.update_layout(font=dict(family="Outfit"), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rain, use_container_width=True)


# ============================================================
# PAGE: MODEL DASHBOARD
# ============================================================
def page_model_dashboard():
    st.markdown(
        """<div class="main-header">
            <h1>📈 Model Dashboard</h1>
            <p>मॉडेल माहिती | Model performance and feature importance</p>
        </div>""",
        unsafe_allow_html=True,
    )

    model_info = predictor.get_model_info()

    # Model status cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌾 Yield Model")
        ym = model_info.get("yield_model", {})
        metrics = ym.get("metrics", {})
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{metrics.get("rmse", "N/A")}</div>'
            f'<div class="metric-label">RMSE (kg/ha)</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"- **Type:** {ym.get('type', 'N/A')}")
        st.markdown(f"- **Features:** {ym.get('n_features', 'N/A')}")
        st.markdown(f"- **Training samples:** {ym.get('n_train_samples', 'N/A'):,}")
        if metrics:
            st.json(metrics)

    with col2:
        st.markdown("### 🦠 Disease Model")
        dm = model_info.get("disease_model", {})
        metrics_d = dm.get("metrics", {})
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{metrics_d.get("f1_weighted", "N/A")}</div>'
            f'<div class="metric-label">F1 Score (weighted)</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"- **Type:** {dm.get('type', 'N/A')}")
        st.markdown(f"- **Features:** {dm.get('n_features', 'N/A')}")
        st.markdown(f"- **Training samples:** {dm.get('n_train_samples', 'N/A'):,}")
        if metrics_d:
            st.json(metrics_d)

    # Feature importance from metadata
    st.markdown("---")
    st.markdown("### 🔑 Feature Importance")

    yield_meta_path = PROJECT_ROOT / "models" / "yield_model_metadata.json"
    if yield_meta_path.exists():
        with open(yield_meta_path) as f:
            ymeta = json.load(f)
        top_feats = ymeta.get("top_features", [])
        if top_feats:
            feat_df = pd.DataFrame({
                "Feature": top_feats[:10],
                "Rank": list(range(1, min(11, len(top_feats) + 1))),
            })
            fig = px.bar(
                feat_df, x="Feature", y="Rank",
                title="Top 10 Features (Yield Model)",
                color="Rank",
                color_continuous_scale=["#1B5E20", "#81C784"],
            )
            fig.update_layout(
                font=dict(family="Outfit"),
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Per-division R²
    if yield_meta_path.exists():
        div_r2 = ymeta.get("per_division_r2", {})
        if div_r2:
            st.markdown("### 📊 Per-Division R² (Test Set)")
            r2_df = pd.DataFrame(list(div_r2.items()), columns=["Division", "R²"])
            fig_r2 = px.bar(
                r2_df, x="Division", y="R²", color="R²",
                color_continuous_scale=["#FFCDD2", "#4CAF50"],
                title="R² Score by Division",
            )
            fig_r2.update_layout(font=dict(family="Outfit"), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_r2, use_container_width=True)


# ============================================================
# ROUTER
# ============================================================
if page == "home":
    page_home()
elif page == "yield":
    page_yield_predictor()
elif page == "disease":
    page_disease_scanner()
elif page == "district":
    page_district_insights()
elif page == "division":
    page_division_explorer()
elif page == "model":
    page_model_dashboard()

# Footer
st.markdown(
    '<div class="footer">'
    '🌾 CropGuard ML v1.0 | Maharashtra Agriculture AI Platform | '
    'Built with ❤️ for Indian Farmers<br>'
    'महाराष्ट्रातील शेतकऱ्यांसाठी AI तंत्रज्ञान'
    '</div>',
    unsafe_allow_html=True,
)
