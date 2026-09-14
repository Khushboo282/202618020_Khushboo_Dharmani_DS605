"""Task 2: Model training, comparison, tuning, evaluation"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "airbnb_cleaned.csv")

# Features used by the model (raw, pre-engineered log columns dropped in favor of pipeline-native transforms
# EXCEPT we keep engineered features that aren't simple scaling, e.g. dist_from_center, ratios, flags)
feature_cols = [
    'neighbourhood_group', 'neighbourhood', 'room_type',       # categorical
    'latitude', 'longitude', 'dist_from_center',                # geo
    'minimum_nights', 'number_of_reviews', 'reviews_per_month', # activity (raw; pipeline will log-transform via FunctionTransformer... simplified: use raw, models handle nonlinearity)
    'calculated_host_listings_count', 'availability_365', 'availability_ratio',
    'has_reviews', 'never_reviewed', 'is_multi_listing_host'
]
target_col = 'log_price'  # model trained on log scale

X = df[feature_cols].copy()
y = df[target_col].copy()
y_dollars = df['price'].copy()

X_train, X_test, y_train, y_test, y_train_d, y_test_d = train_test_split(
    X, y, y_dollars, test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

cat_onehot = ['neighbourhood_group', 'room_type']
cat_target = ['neighbourhood']
num_cols = [c for c in feature_cols if c not in cat_onehot + cat_target]

preprocessor = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_onehot),
    ('target_enc', TargetEncoder(random_state=42), cat_target),
    ('num', StandardScaler(), num_cols),
])

def eval_model(name, pipe, X_tr, y_tr_log, X_te, y_te_log, y_te_dollars):
    pipe.fit(X_tr, y_tr_log)
    pred_log_tr = pipe.predict(X_tr)
    pred_log_te = pipe.predict(X_te)
    # invert log for $ metrics on test set
    pred_dollars_te = np.expm1(pred_log_te)
    pred_dollars_te = np.clip(pred_dollars_te, 0, None)

    r2_tr = r2_score(y_tr_log, pred_log_tr)
    r2_te = r2_score(y_te_log, pred_log_te)
    mae_d = mean_absolute_error(y_te_dollars, pred_dollars_te)
    rmse_d = np.sqrt(mean_squared_error(y_te_dollars, pred_dollars_te))
    rmse_log = np.sqrt(mean_squared_error(y_te_log, pred_log_te))

    print(f"{name:28s} | R2 train={r2_tr:.4f} | R2 test={r2_te:.4f} | RMSE(log)={rmse_log:.4f} | MAE($)={mae_d:.2f} | RMSE($)={rmse_d:.2f}")
    return {'name': name, 'r2_train': r2_tr, 'r2_test': r2_te, 'rmse_log': rmse_log, 'mae_dollars': mae_d, 'rmse_dollars': rmse_d}

results = []

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.08, random_state=42),
}

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    res = eval_model(name, pipe, X_train, y_train, X_test, y_test, y_test_d)
    results.append(res)

results_df = pd.DataFrame(results).sort_values('r2_test', ascending=False)
print("\n=== Model Comparison (sorted by test R2) ===")
print(results_df.to_string(index=False))
results_df.to_csv(ROOT / "assets" / "model_comparison_baseline.csv", index=False)
