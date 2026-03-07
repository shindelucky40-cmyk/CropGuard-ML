"""
CropGuard ML — Feature Engineering Pipeline
Computes all derived features and prepares data for model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from pathlib import Path

from src.constants import (
    CROP_WATER_REQUIREMENT, IRRIGATION_EFFICIENCY, CROP_OPTIMAL_PH,
    CROP_AVG_YIELD, ALL_SOIL_TYPES, MAJOR_CROPS, IRRIGATION_TYPES,
    SEED_VARIETIES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    WEATHER_FEATURES, NDVI_FEATURES
)


def compute_rainfall_adequacy(df: pd.DataFrame) -> pd.Series:
    """Actual rainfall / crop water requirement ratio."""
    water_req = df["crop_name"].map(CROP_WATER_REQUIREMENT).fillna(500)
    return (df["rainfall_mm_seasonal"] / water_req).round(3)


def compute_growing_degree_days(df: pd.DataFrame, t_base: float = 10.0) -> pd.Series:
    """GDD = sum((Tmax + Tmin)/2 - T_base) over ~120 growing days."""
    daily_gdd = ((df["temp_max_avg_C"] + df["temp_min_avg_C"]) / 2) - t_base
    daily_gdd = daily_gdd.clip(lower=0)
    # Approximate: multiply by ~120 growing season days
    return (daily_gdd * 120).round(1)


def compute_vpd(df: pd.DataFrame) -> pd.Series:
    """
    Vapour Pressure Deficit (kPa).
    VPD = (1 - RH/100) × 0.6108 × exp(17.27×T / (T+237.3))
    """
    t = df["temp_max_avg_C"]
    rh = df["humidity_avg_pct"]
    sat_vp = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
    vpd = (1 - rh / 100) * sat_vp
    return vpd.round(3)


def compute_soil_crop_suitability(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based suitability score (0–1).
    Checks soil pH vs crop optimal range + soil type compatibility.
    """
    scores = []
    for _, row in df.iterrows():
        score = 0.5  # base
        crop = row["crop_name"]
        ph = row["soil_pH"]

        # pH suitability
        opt = CROP_OPTIMAL_PH.get(crop, (6.0, 7.5))
        if opt[0] <= ph <= opt[1]:
            score += 0.3
        elif abs(ph - np.mean(opt)) < 1.5:
            score += 0.1
        else:
            score -= 0.2

        # Organic carbon bonus
        if row["organic_carbon_pct"] > 0.6:
            score += 0.1

        # Soil moisture
        if row["soil_moisture_pct"] > 20:
            score += 0.1

        scores.append(max(0, min(1, round(score, 2))))

    return pd.Series(scores, index=df.index)


def compute_irrigation_efficiency(df: pd.DataFrame) -> pd.Series:
    """Map irrigation type to efficiency score."""
    return df["irrigation_type"].map(IRRIGATION_EFFICIENCY).fillna(0.5)


def compute_pest_pressure_index(df: pd.DataFrame) -> pd.Series:
    """Combined pest pressure: humidity_streak × temp_disease_window / 30."""
    return ((df["humidity_streak_days"] * df["temp_disease_window"]) / 30).clip(0, 20).round(3)


def compute_yield_lag(df: pd.DataFrame) -> pd.Series:
    """Previous year yield for same (district, crop)."""
    df_sorted = df.sort_values(["district", "crop_name", "year"])
    lag = df_sorted.groupby(["district", "crop_name"])["yield_kg_per_hectare"].shift(1)
    return lag.reindex(df.index)


def compute_ndvi_anomaly(df: pd.DataFrame) -> pd.Series:
    """NDVI peak minus historical mean NDVI for the district."""
    hist_mean = df.groupby("district")["ndvi_peak"].transform("mean")
    return (df["ndvi_peak"] - hist_mean).round(3)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df = df.copy()

    # Derived features
    df["rainfall_adequacy_ratio"] = compute_rainfall_adequacy(df)
    df["growing_degree_days"] = compute_growing_degree_days(df)
    df["vapour_pressure_deficit"] = compute_vpd(df)
    df["soil_crop_suitability_score"] = compute_soil_crop_suitability(df)
    df["irrigation_efficiency_score"] = compute_irrigation_efficiency(df)
    df["pest_pressure_index"] = compute_pest_pressure_index(df)
    df["yield_lag_1yr"] = compute_yield_lag(df)
    df["ndvi_anomaly"] = compute_ndvi_anomaly(df)

    # Fill NaN in lag feature (first year has no lag)
    df["yield_lag_1yr"] = df["yield_lag_1yr"].fillna(
        df["crop_name"].map(CROP_AVG_YIELD)
    )

    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True,
                     encoders: dict = None, scalers: dict = None):
    """
    Encode categoricals and scale numerical features.
    Returns (df_encoded, encoders, scalers) for reuse on val/test.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
    if scalers is None:
        scalers = {}

    # --- Label Encoding ---
    label_cols = ["crop_name", "irrigation_type", "seed_variety",
                  "previous_crop", "season"]
    for col in label_cols:
        if fit:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col + "_encoded"] = df[col].apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

    # --- OneHot Encoding ---
    onehot_cols = ["soil_type", "division"]
    for col in onehot_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)

    # --- Standard Scaling (weather features) ---
    weather_to_scale = [f for f in WEATHER_FEATURES if f in df.columns]
    if fit:
        ss = StandardScaler()
        df[weather_to_scale] = ss.fit_transform(df[weather_to_scale])
        scalers["weather"] = ss
    else:
        df[weather_to_scale] = scalers["weather"].transform(df[weather_to_scale])

    # --- MinMax Scaling (NDVI features, bounded 0-1) ---
    ndvi_to_scale = [f for f in NDVI_FEATURES if f in df.columns]
    if fit:
        mms = MinMaxScaler()
        df[ndvi_to_scale] = mms.fit_transform(df[ndvi_to_scale])
        scalers["ndvi"] = mms
    else:
        df[ndvi_to_scale] = scalers["ndvi"].transform(df[ndvi_to_scale])

    return df, encoders, scalers


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get all feature columns for model training (excludes targets and identifiers)."""
    exclude = [
        "yield_kg_per_hectare", "disease_risk_score", "disease_risk_label",
        "district", "division", "crop_name", "season", "year",
        "soil_type", "irrigation_type", "seed_variety", "previous_crop",
        "scenario"
    ]
    return [c for c in df.columns if c not in exclude]


def prepare_training_data(data_dir: str = "data"):
    """Full pipeline: load → engineer → encode → split → save."""
    base = Path(data_dir)

    print("⚙️  Running feature engineering pipeline...")

    # Load processed data
    df = pd.read_csv(base / "processed" / "mh_cleaned_merged.csv")
    print(f"   Loaded {len(df)} records")

    # Engineer features
    df = engineer_features(df)
    print(f"   Engineered {len(df.columns)} total columns")

    # Split by year FIRST (before encoding to avoid data leakage)
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    val = df[df["year"].isin(VAL_YEARS)].copy()
    test = df[df["year"].isin(TEST_YEARS)].copy()

    # Encode & Scale (fit on train only)
    train, encoders, scalers = encode_and_scale(train, fit=True)
    val, _, _ = encode_and_scale(val, fit=False, encoders=encoders, scalers=scalers)
    test, _, _ = encode_and_scale(test, fit=False, encoders=encoders, scalers=scalers)

    # Save
    features_dir = base / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(features_dir / "train_engineered.csv", index=False)
    val.to_csv(features_dir / "val_engineered.csv", index=False)
    test.to_csv(features_dir / "test_engineered.csv", index=False)

    print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("✅ Feature engineering complete!")

    return train, val, test, encoders, scalers


if __name__ == "__main__":
    prepare_training_data()
