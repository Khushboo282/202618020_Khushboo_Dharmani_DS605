"""Fit the documented final model and save artifacts used by the Streamlit app.

Uses the tuned HistGradientBoosting hyperparameters from the lab notebook
(max_iter=450, max_depth=8, learning_rate=0.05, min_samples_leaf=20).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
DATA_PATH = ROOT / "data" / "airbnb_cleaned.csv"

FEATURE_COLS = [
    "neighbourhood_group",
    "neighbourhood",
    "room_type",
    "latitude",
    "longitude",
    "dist_from_center",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "availability_ratio",
    "has_reviews",
    "never_reviewed",
    "is_multi_listing_host",
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].copy()
    y = df["log_price"].copy()
    y_dollars = df["price"].copy()

    X_train, X_test, y_train, y_test, y_train_d, y_test_d = train_test_split(
        X, y, y_dollars, test_size=0.2, random_state=42
    )

    cat_onehot = ["neighbourhood_group", "room_type"]
    cat_target = ["neighbourhood"]
    num_cols = [c for c in FEATURE_COLS if c not in cat_onehot + cat_target]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_onehot,
            ),
            ("target_enc", TargetEncoder(random_state=42), cat_target),
            ("num", StandardScaler(), num_cols),
        ]
    )

    best_params = {
        "max_iter": 450,
        "max_depth": 8,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "random_state": 42,
    }

    pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", HistGradientBoostingRegressor(**best_params)),
        ]
    )
    pipe.fit(X_train, y_train)

    pred_log_tr = pipe.predict(X_train)
    pred_log_te = pipe.predict(X_test)
    pred_d_te = np.clip(np.expm1(pred_log_te), 0, None)
    r2_tr = r2_score(y_train, pred_log_tr)
    r2_te = r2_score(y_test, pred_log_te)
    mae_d = float(mean_absolute_error(y_test_d, pred_d_te))
    rmse_d = float(np.sqrt(mean_squared_error(y_test_d, pred_d_te)))

    print(
        f"HistGB | R2 train={r2_tr:.4f} | R2 test={r2_te:.4f} | "
        f"MAE($)={mae_d:.2f} | RMSE($)={rmse_d:.2f}"
    )

    APP_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, APP_DIR / "airbnb_price_pipeline.pkl")
    joblib.dump(
        {
            "feature_cols": FEATURE_COLS,
            "best_model": "HistGradientBoosting",
            "best_params": best_params,
            "r2_test": float(r2_te),
            "mae_dollars": mae_d,
            "rmse_dollars": rmse_d,
        },
        APP_DIR / "model_meta.pkl",
    )
    print(f"Saved {APP_DIR / 'airbnb_price_pipeline.pkl'}")
    print(f"Saved {APP_DIR / 'model_meta.pkl'}")


if __name__ == "__main__":
    main()
