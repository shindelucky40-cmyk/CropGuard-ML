"""
CropGuard ML — Yield Prediction Model Training
XGBoost Regressor with Optuna hyperparameter tuning.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


def get_yield_features(df: pd.DataFrame) -> list:
    """Get feature columns for yield model."""
    exclude = [
        "yield_kg_per_hectare", "disease_risk_score", "disease_risk_label",
        "district", "division", "crop_name", "season", "year",
        "soil_type", "irrigation_type", "seed_variety", "previous_crop",
        "scenario"
    ]
    return [c for c in df.columns if c not in exclude and not df[c].dtype == "object"]


def train_yield_model(data_dir: str = "data", model_dir: str = "models",
                      use_optuna: bool = True, n_trials: int = 50):
    """
    Train XGBoost yield regression model.
    """
    base = Path(data_dir)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    print("🌾 Training Yield Prediction Model...")

    # Load engineered data
    train = pd.read_csv(base / "features" / "train_engineered.csv")
    val = pd.read_csv(base / "features" / "val_engineered.csv")
    test = pd.read_csv(base / "features" / "test_engineered.csv")

    feature_cols = get_yield_features(train)
    # Ensure same columns
    common_cols = [c for c in feature_cols if c in val.columns and c in test.columns]
    feature_cols = common_cols

    X_train = train[feature_cols].fillna(0)
    y_train = train["yield_kg_per_hectare"]
    X_val = val[feature_cols].fillna(0)
    y_val = val["yield_kg_per_hectare"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["yield_kg_per_hectare"]

    print(f"   Features: {len(feature_cols)}")
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    if use_optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": 42,
            }

            model = XGBRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            return rmse

        print(f"   Running Optuna ({n_trials} trials)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params["random_state"] = 42
        print(f"   Best RMSE (val): {study.best_value:.1f}")
    else:
        best_params = {
            "n_estimators": 800, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
            "min_child_weight": 3, "random_state": 42,
        }

    # Train final model
    print("   Training final model...")
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    # Evaluate
    metrics = {}
    for name, X, y in [("train", X_train, y_train),
                        ("val", X_val, y_val),
                        ("test", X_test, y_test)]:
        pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mape = mean_absolute_percentage_error(y, pred) * 100
        r2 = r2_score(y, pred)
        metrics[name] = {"rmse": round(rmse, 1), "mape": round(mape, 2), "r2": round(r2, 4)}
        print(f"   {name.upper():6s} — RMSE: {rmse:.1f} | MAPE: {mape:.1f}% | R²: {r2:.4f}")

    # Per-division R²
    if "division" not in test.columns:
        test_with_div = pd.read_csv(base / "features" / "test_engineered.csv")
    else:
        test_with_div = test
    test_pred = model.predict(X_test)
    div_r2 = {}
    for div in test_with_div["division"].unique() if "division" in test_with_div.columns else []:
        mask = test_with_div["division"] == div
        if mask.sum() > 5:
            div_r2[div] = round(r2_score(y_test[mask], test_pred[mask]), 4)
    if div_r2:
        print("   Per-division R² (test):")
        for d, r in div_r2.items():
            print(f"     {d}: {r}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    print("   Top 15 features:")
    for feat, imp in top_features:
        print(f"     {feat}: {imp:.4f}")

    # Save model + metadata
    joblib.dump(model, model_path / "yield_model.joblib")
    metadata = {
        "model_type": "XGBRegressor",
        "params": best_params,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "top_features": [f[0] for f in top_features],
        "per_division_r2": div_r2,
        "n_train_samples": len(X_train),
    }
    with open(model_path / "yield_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"   Model saved: {model_path / 'yield_model.joblib'}")
    print("✅ Yield model training complete!")

    return model, metrics


if __name__ == "__main__":
    train_yield_model(use_optuna=True, n_trials=50)
