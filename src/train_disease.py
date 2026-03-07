"""
CropGuard ML — Disease Risk Classification Model Training
XGBoost Classifier with SMOTE for class imbalance, per-crop models.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, f1_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


def get_disease_features(df: pd.DataFrame) -> list:
    """Get feature columns for disease model."""
    exclude = [
        "yield_kg_per_hectare", "disease_risk_score", "disease_risk_label",
        "district", "division", "crop_name", "season", "year",
        "soil_type", "irrigation_type", "seed_variety", "previous_crop",
        "scenario"
    ]
    return [c for c in df.columns if c not in exclude and not df[c].dtype == "object"]


def train_disease_model(data_dir: str = "data", model_dir: str = "models",
                        use_optuna: bool = True, n_trials: int = 30):
    """
    Train XGBoost disease risk classification model.
    Creates a unified classifier across all crops.
    """
    base = Path(data_dir)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    print("🦠 Training Disease Risk Classification Model...")

    # Load engineered data
    train = pd.read_csv(base / "features" / "train_engineered.csv")
    val = pd.read_csv(base / "features" / "val_engineered.csv")
    test = pd.read_csv(base / "features" / "test_engineered.csv")

    feature_cols = get_disease_features(train)
    common_cols = [c for c in feature_cols if c in val.columns and c in test.columns]
    feature_cols = common_cols

    # Remove yield from disease features (no data leakage)
    feature_cols = [c for c in feature_cols if "yield" not in c.lower() or "lag" in c.lower()]

    # Encode target labels
    label_encoder = LabelEncoder()
    label_order = ["Low", "Medium", "High", "Critical"]
    label_encoder.fit(label_order)

    X_train = train[feature_cols].fillna(0)
    y_train = label_encoder.transform(train["disease_risk_label"])
    X_val = val[feature_cols].fillna(0)
    y_val = label_encoder.transform(val["disease_risk_label"])
    X_test = test[feature_cols].fillna(0)
    y_test = label_encoder.transform(test["disease_risk_label"])

    print(f"   Features: {len(feature_cols)}")
    print(f"   Class distribution (train):")
    for cls, count in zip(*np.unique(y_train, return_counts=True)):
        print(f"     {label_order[cls]}: {count}")

    # Apply SMOTE to balance classes
    print("   Applying SMOTE for class balancing...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train)) - 1))
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE: {len(X_train_res)} samples")
    except ValueError:
        print("   SMOTE not applicable (too few samples in a class). Using original data.")
        X_train_res, y_train_res = X_train, y_train

    if use_optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
            }

            model = XGBClassifier(**params)
            model.fit(X_train_res, y_train_res,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

            pred = model.predict(X_val)
            f1 = f1_score(y_val, pred, average="weighted")
            return f1

        print(f"   Running Optuna ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["use_label_encoder"] = False
        best_params["eval_metric"] = "mlogloss"
        print(f"   Best F1 (val): {study.best_value:.4f}")
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 5, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 1.0, "reg_lambda": 3.0,
            "min_child_weight": 3, "random_state": 42,
            "use_label_encoder": False, "eval_metric": "mlogloss",
        }

    # Train final model
    print("   Training final model...")
    model = XGBClassifier(**best_params)
    model.fit(X_train_res, y_train_res,
              eval_set=[(X_val, y_val)],
              verbose=False)

    # Evaluate
    metrics = {}
    for name, X, y in [("train", X_train, y_train),
                        ("val", X_val, y_val),
                        ("test", X_test, y_test)]:
        pred = model.predict(X)
        f1 = f1_score(y, pred, average="weighted")
        # Recall for Critical class (index 3)
        critical_mask = y == 3
        if critical_mask.sum() > 0:
            recall_critical = recall_score(y == 3, pred == 3, zero_division=0)
        else:
            recall_critical = 0
        metrics[name] = {
            "f1_weighted": round(f1, 4),
            "recall_critical": round(recall_critical, 4)
        }
        print(f"   {name.upper():6s} — F1: {f1:.4f} | Recall(Critical): {recall_critical:.4f}")

    # Full classification report on test set
    test_pred = model.predict(X_test)
    print("\n   Classification Report (Test):")
    print(classification_report(y_test, test_pred,
                                 target_names=label_order))

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("   Top 10 disease risk features:")
    for feat, imp in top_features:
        print(f"     {feat}: {imp:.4f}")

    # Save
    joblib.dump(model, model_path / "disease_model.joblib")
    joblib.dump(label_encoder, model_path / "disease_label_encoder.joblib")

    metadata = {
        "model_type": "XGBClassifier",
        "params": best_params,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "label_order": label_order,
        "top_features": [f[0] for f in top_features],
        "n_train_samples": len(X_train_res),
    }
    with open(model_path / "disease_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"   Model saved: {model_path / 'disease_model.joblib'}")
    print("✅ Disease model training complete!")

    return model, metrics


if __name__ == "__main__":
    train_disease_model(use_optuna=True, n_trials=30)
