"""
CropGuard ML — Master Run Script
Loads raw data, engineers features, trains models — all in one command.
"""

import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("  CropGuard ML -- Full Pipeline Runner")
    print("  Maharashtra Agriculture AI Platform")
    print("=" * 60)

    # Step 1: Verify raw data exists
    print("\n[1/3] Verifying raw dataset...")
    raw_path = PROJECT_ROOT / "data" / "raw" / "maharashtra_agri_survey_2010_2024.csv"
    if not raw_path.exists():
        print(f"  ERROR: Raw dataset not found at {raw_path}")
        print("  Please place the Maharashtra agriculture survey CSV in data/raw/")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(raw_path)
    print(f"  Loaded: {len(df)} records, {len(df.columns)} features")
    print(f"  Districts: {df['district'].nunique()} | Crops: {df['crop_name'].nunique()}")
    print(f"  Year range: {df['year'].min()} - {df['year'].max()}")

    # Save to processed
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "mh_cleaned_merged.csv", index=False)
    print("  Saved cleaned data to data/processed/mh_cleaned_merged.csv")

    # Step 2: Feature engineering
    print("\n[2/3] Running feature engineering pipeline...")
    from src.features import prepare_training_data
    prepare_training_data()

    # Step 3: Train yield model
    print("\n[3a/3] Training yield prediction model...")
    from src.train_yield import train_yield_model
    train_yield_model(use_optuna=True, n_trials=50)

    # Step 3b: Train disease model
    print("\n[3b/3] Training disease risk model...")
    from src.train_disease import train_disease_model
    train_disease_model(use_optuna=True, n_trials=30)

    print("\n" + "=" * 60)
    print("  CropGuard ML pipeline complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start API:        python -m uvicorn api.main:app --reload --port 8000")
    print("  2. Start Dashboard:  python -m streamlit run app/streamlit_app.py")
    print("  3. Run tests:        python -m pytest tests/ -v")
    print("  4. API docs:         http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
