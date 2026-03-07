"""
CropGuard ML — Maharashtra Agricultural Constants
All district, crop, soil, and climate data for Maharashtra's 36 districts.
"""

# ============================================================
# MAHARASHTRA DISTRICTS → DIVISIONS MAPPING
# ============================================================
DIVISIONS = {
    "Konkan": [
        "Mumbai City", "Mumbai Suburban", "Thane", "Palghar",
        "Raigad", "Ratnagiri", "Sindhudurg"
    ],
    "Nashik": [
        "Nashik", "Ahmednagar", "Dhule", "Nandurbar", "Jalgaon"
    ],
    "Pune": [
        "Pune", "Satara", "Sangli", "Solapur", "Kolhapur"
    ],
    "Aurangabad": [
        "Chhatrapati Sambhajinagar", "Jalna", "Beed", "Latur",
        "Dharashiv", "Nanded", "Parbhani", "Hingoli"
    ],
    "Amravati": [
        "Amravati", "Akola", "Washim", "Buldhana", "Yavatmal"
    ],
    "Nagpur": [
        "Nagpur", "Wardha", "Chandrapur", "Gadchiroli",
        "Gondia", "Bhandara"
    ],
}

# Flat list of all 36 districts
ALL_DISTRICTS = []
for div, dists in DIVISIONS.items():
    ALL_DISTRICTS.extend(dists)

DISTRICT_TO_DIVISION = {}
for div, dists in DIVISIONS.items():
    for d in dists:
        DISTRICT_TO_DIVISION[d] = div

# Renamed districts mapping (old → new)
DISTRICT_RENAMES = {
    "Aurangabad": "Chhatrapati Sambhajinagar",
    "Osmanabad": "Dharashiv",
}

# ============================================================
# CROPS & SEASONS
# ============================================================
SEASONS = {
    "Kharif": {"marathi": "खरीप", "months": "June–October",
               "crops": ["Soybean", "Cotton", "Rice", "Tur Dal", "Maize", "Jowar"]},
    "Rabi": {"marathi": "रबी", "months": "November–March",
             "crops": ["Wheat", "Jowar", "Gram", "Onion", "Sunflower"]},
    "Summer": {"marathi": "उन्हाळी", "months": "March–June",
               "crops": ["Watermelon", "Cucumber", "Fodder", "Vegetables"]},
    "Year-Round": {"marathi": "वार्षिक", "months": "All year",
                   "crops": ["Sugarcane", "Grapes", "Banana"]},
}

MAJOR_CROPS = [
    "Soybean", "Cotton", "Rice", "Onion",
    "Sugarcane", "Grapes", "Tur Dal", "Jowar"
]

# Crops grown per division (primary)
DIVISION_CROPS = {
    "Konkan": ["Rice", "Coconut", "Cashew", "Mango"],
    "Nashik": ["Grapes", "Onion", "Wheat", "Maize", "Banana"],
    "Pune": ["Sugarcane", "Soybean", "Jowar", "Turmeric"],
    "Aurangabad": ["Soybean", "Cotton", "Tur Dal", "Jowar"],
    "Amravati": ["Cotton", "Soybean", "Orange", "Wheat"],
    "Nagpur": ["Rice", "Cotton", "Wheat", "Soybean"],
}

# Crop → Season mapping
CROP_SEASON = {
    "Soybean": "Kharif", "Cotton": "Kharif", "Rice": "Kharif",
    "Tur Dal": "Kharif", "Maize": "Kharif", "Jowar": "Kharif",
    "Wheat": "Rabi", "Gram": "Rabi", "Onion": "Rabi", "Sunflower": "Rabi",
    "Sugarcane": "Year-Round", "Grapes": "Year-Round", "Banana": "Year-Round",
}

# ============================================================
# CROP-DISEASE PAIRS
# ============================================================
CROP_DISEASES = {
    "Soybean": ["Yellow Mosaic Virus", "Bacterial Pustule", "Pod Borer"],
    "Cotton": ["Pink Bollworm", "Bacterial Blight", "Root Rot"],
    "Sugarcane": ["Red Rot", "Wilt", "Smut", "Mosaic Virus"],
    "Onion": ["Purple Blotch", "Stemphylium", "Downy Mildew"],
    "Rice": ["Blast", "Brown Planthopper", "Sheath Blight"],
    "Grapes": ["Downy Mildew", "Botrytis", "Powdery Mildew"],
    "Tur Dal": ["Fusarium Wilt", "Sterility Mosaic", "Phytophthora Blight"],
    "Jowar": ["Anthracnose", "Downy Mildew", "Grain Mold"],
}

# Marathi crop names
CROP_MARATHI = {
    "Soybean": "सोयाबीन", "Cotton": "कापूस", "Rice": "भात/तांदूळ",
    "Onion": "कांदा", "Sugarcane": "ऊस", "Grapes": "द्राक्षे",
    "Tur Dal": "तूर", "Jowar": "ज्वारी", "Wheat": "गहू",
    "Maize": "मका", "Gram": "हरभरा", "Banana": "केळी",
}

# Marathi disease risk labels
RISK_LABELS_MARATHI = {
    "Low": "कमी धोका",
    "Medium": "मध्यम धोका",
    "High": "जास्त धोका",
    "Critical": "अत्यंत धोका",
}

# ============================================================
# SOIL TYPES PER DIVISION
# ============================================================
DIVISION_SOIL_TYPES = {
    "Konkan": ["Laterite", "Alluvial"],
    "Nashik": ["Medium Black", "Red"],
    "Pune": ["Deep Black Cotton", "Medium Black"],
    "Aurangabad": ["Medium Black", "Deep Black Cotton"],
    "Amravati": ["Deep Black Cotton", "Medium Black"],
    "Nagpur": ["Red Laterite", "Alluvial", "Deep Black Cotton"],
}

ALL_SOIL_TYPES = [
    "Deep Black Cotton", "Medium Black", "Red",
    "Laterite", "Alluvial", "Red Laterite"
]

# ============================================================
# CLIMATE BASELINES (district-level Long Period Averages)
# ============================================================
# Annual rainfall LPA in mm per division (approximate)
DIVISION_RAINFALL_LPA = {
    "Konkan": 2800, "Nashik": 700, "Pune": 850,
    "Aurangabad": 650, "Amravati": 800, "Nagpur": 1050,
}

# Temperature ranges °C (min_avg, max_avg) per division
DIVISION_TEMP_RANGE = {
    "Konkan": (22, 33), "Nashik": (18, 36),
    "Pune": (17, 35), "Aurangabad": (19, 38),
    "Amravati": (18, 39), "Nagpur": (17, 40),
}

# Humidity % range (low, high) per division
DIVISION_HUMIDITY_RANGE = {
    "Konkan": (65, 90), "Nashik": (35, 75),
    "Pune": (35, 80), "Aurangabad": (30, 70),
    "Amravati": (30, 70), "Nagpur": (30, 75),
}

# ============================================================
# AVERAGE YIELDS (kg/hectare) — baseline references
# ============================================================
CROP_AVG_YIELD = {
    "Soybean": 1200, "Cotton": 400, "Rice": 2500,
    "Onion": 18000, "Sugarcane": 80000, "Grapes": 25000,
    "Tur Dal": 800, "Jowar": 1000, "Wheat": 2000,
    "Maize": 2500, "Gram": 900, "Banana": 40000,
}

# Water requirement per crop (mm per season)
CROP_WATER_REQUIREMENT = {
    "Soybean": 450, "Cotton": 700, "Rice": 1200, "Onion": 400,
    "Sugarcane": 2000, "Grapes": 600, "Tur Dal": 400, "Jowar": 400,
    "Wheat": 350, "Maize": 500, "Gram": 300, "Banana": 1800,
}

# Disease temperature windows (°C range optimal for pathogen)
DISEASE_TEMP_WINDOWS = {
    "Yellow Mosaic Virus": (25, 35), "Bacterial Pustule": (25, 32),
    "Pink Bollworm": (25, 35), "Bacterial Blight": (25, 35),
    "Red Rot": (28, 36), "Wilt": (25, 33),
    "Purple Blotch": (20, 30), "Downy Mildew": (18, 25),
    "Blast": (20, 28), "Sheath Blight": (25, 32),
    "Botrytis": (15, 25), "Powdery Mildew": (20, 28),
    "Fusarium Wilt": (25, 33), "Sterility Mosaic": (25, 35),
    "Anthracnose": (25, 30), "Grain Mold": (25, 32),
}

# Irrigation efficiency scores
IRRIGATION_EFFICIENCY = {
    "drip": 1.0, "sprinkler": 0.85,
    "furrow": 0.65, "canal": 0.60, "rainfed": 0.40,
}

IRRIGATION_TYPES = list(IRRIGATION_EFFICIENCY.keys())
SEED_VARIETIES = ["HYV", "Desi", "BT", "Hybrid"]

# Soil pH optimal range per crop
CROP_OPTIMAL_PH = {
    "Soybean": (6.0, 7.5), "Cotton": (6.5, 8.0), "Rice": (5.5, 7.0),
    "Onion": (6.0, 7.5), "Sugarcane": (6.0, 7.5), "Grapes": (6.5, 7.5),
    "Tur Dal": (6.5, 7.5), "Jowar": (6.0, 7.5), "Wheat": (6.0, 7.5),
}

# ============================================================
# FEATURE CONSTANTS
# ============================================================
WEATHER_FEATURES = [
    "rainfall_mm_seasonal", "temp_max_avg_C", "temp_min_avg_C",
    "humidity_avg_pct", "dry_spell_days", "rainfall_deviation_pct",
    "onset_monsoon_deviation_days", "fog_days"
]

SOIL_FEATURES = [
    "soil_type", "soil_pH", "organic_carbon_pct",
    "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha",
    "soil_moisture_pct", "ec_dS_m"
]

CROP_MGMT_FEATURES = [
    "crop_name", "sowing_date_deviation", "irrigation_type",
    "seed_variety", "fertilizer_kg_ha", "pesticide_applications",
    "previous_crop"
]

NDVI_FEATURES = [
    "ndvi_sowing", "ndvi_peak", "ndvi_delta", "ndvi_stress_weeks"
]

DISEASE_FEATURES = [
    "humidity_streak_days", "temp_disease_window",
    "prev_season_disease", "pest_alert_issued"
]

ADMIN_FEATURES = ["district", "division", "season", "year"]

# Target variables
TARGET_YIELD = "yield_kg_per_hectare"
TARGET_DISEASE_SCORE = "disease_risk_score"
TARGET_DISEASE_LABEL = "disease_risk_label"

# Train/Val/Test year split (temporal, never random)
TRAIN_YEARS = list(range(2010, 2021))   # 2010–2020
VAL_YEARS = list(range(2021, 2023))     # 2021–2022
TEST_YEARS = list(range(2023, 2025))    # 2023–2024

# Feature columns used by Model 1 (Yield)
YIELD_FEATURE_COLS = (
    WEATHER_FEATURES + SOIL_FEATURES[1:] +  # exclude soil_type (encoded separately)
    CROP_MGMT_FEATURES[1:] +  # exclude crop_name (encoded separately)
    NDVI_FEATURES +
    [
        "rainfall_adequacy_ratio", "growing_degree_days",
        "vapour_pressure_deficit", "soil_crop_suitability_score",
        "irrigation_efficiency_score", "pest_pressure_index",
        "yield_lag_1yr", "ndvi_anomaly"
    ]
)

# Disease risk thresholds
RISK_THRESHOLDS = {
    "Low": (0.0, 0.25),
    "Medium": (0.25, 0.50),
    "High": (0.50, 0.75),
    "Critical": (0.75, 1.0),
}
