"""Task 2 (cont.): Hyperparameter tuning of top candidates + overfitting check.
Note: sandbox has 1 CPU core, so RandomForest (slow per-fit) is tuned via a lighter
manual grid on a validation split, while HistGradientBoosting (fast per-fit) gets a
full 5-fold RandomizedSearchCV."""
import pandas as pd, numpy as np, time, itertools
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('/home/claude/airbnb-price-prediction/data/airbnb_cleaned.csv')

feature_cols = [
    'neighbourhood_group', 'neighbourhood', 'room_type',
    'latitude', 'longitude', 'dist_from_center',
    'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365', 'availability_ratio',
    'has_reviews', 'never_reviewed', 'is_multi_listing_host'
]
target_col = 'log_price'

X = df[feature_cols].copy()
y = df[target_col].copy()
y_dollars = df['price'].copy()

X_train, X_test, y_train, y_test, y_train_d, y_test_d = train_test_split(
    X, y, y_dollars, test_size=0.2, random_state=42
)
# further split train -> sub_train/val for quick RF tuning
X_subtr, X_val, y_subtr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

cat_onehot = ['neighbourhood_group', 'room_type']
cat_target = ['neighbourhood']
num_cols = [c for c in feature_cols if c not in cat_onehot + cat_target]

def make_preprocessor():
    return ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_onehot),
        ('target_enc', TargetEncoder(random_state=42), cat_target),
        ('num', StandardScaler(), num_cols),
    ])

# ---------------- RandomForest: light manual grid on validation split ----------------
rf_grid = [
    {'n_estimators': 100, 'max_depth': 8,  'min_samples_leaf': 4},
    {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 4},
    {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 6},
    {'n_estimators': 150, 'max_depth': 14, 'min_samples_leaf': 2},
]
print("=== RandomForest manual grid (validation split) ===")
best_rf_params, best_rf_val_r2 = None, -np.inf
t0 = time.time()
for params in rf_grid:
    pipe = Pipeline([('prep', make_preprocessor()),
                      ('model', RandomForestRegressor(random_state=42, n_jobs=1, **params))])
    pipe.fit(X_subtr, y_subtr)
    val_r2 = r2_score(y_val, pipe.predict(X_val))
    tr_r2 = r2_score(y_subtr, pipe.predict(X_subtr))
    print(f"{params} -> train R2={tr_r2:.4f} val R2={val_r2:.4f}")
    if val_r2 > best_rf_val_r2:
        best_rf_val_r2, best_rf_params = val_r2, params
print(f"RF grid search done in {time.time()-t0:.1f}s | best params={best_rf_params} val R2={best_rf_val_r2:.4f}")

# refit best RF on full training set
rf_final = Pipeline([('prep', make_preprocessor()),
                      ('model', RandomForestRegressor(random_state=42, n_jobs=1, **best_rf_params))])
t0=time.time()
rf_final.fit(X_train, y_train)
print(f"RF refit on full train set: {time.time()-t0:.1f}s")

# ---------------- HistGradientBoosting: full RandomizedSearchCV (fast) ----------------
print("\n=== HistGradientBoosting RandomizedSearchCV (5-fold) ===")
hgb_pipe = Pipeline([('prep', make_preprocessor()), ('model', HistGradientBoostingRegressor(random_state=42))])
hgb_param_dist = {
    'model__max_iter': [150, 250, 350, 450],
    'model__max_depth': [4, 6, 8, None],
    'model__learning_rate': [0.03, 0.05, 0.08, 0.1],
    'model__l2_regularization': [0.0, 0.5, 1.0],
    'model__min_samples_leaf': [10, 20, 30],
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
hgb_search = RandomizedSearchCV(hgb_pipe, hgb_param_dist, n_iter=20, cv=cv, scoring='r2', random_state=42, n_jobs=1, verbose=0)
t0 = time.time()
hgb_search.fit(X_train, y_train)
print(f"HGB search done in {time.time()-t0:.1f}s | best CV R2={hgb_search.best_score_:.4f}")
print("Best HGB params:", hgb_search.best_params_)

# ---------------- Final comparison on held-out test set ----------------
def evaluate(name, pipe):
    pred_log_tr = pipe.predict(X_train)
    pred_log_te = pipe.predict(X_test)
    pred_d_te = np.clip(np.expm1(pred_log_te), 0, None)
    r2_tr = r2_score(y_train, pred_log_tr)
    r2_te = r2_score(y_test, pred_log_te)
    mae_d = mean_absolute_error(y_test_d, pred_d_te)
    rmse_d = np.sqrt(mean_squared_error(y_test_d, pred_d_te))
    rmse_log = np.sqrt(mean_squared_error(y_test, pred_log_te))
    print(f"{name:20s} | R2 train={r2_tr:.4f} | R2 test={r2_te:.4f} | gap={r2_tr-r2_te:.4f} | RMSE(log)={rmse_log:.4f} | MAE($)={mae_d:.2f} | RMSE($)={rmse_d:.2f}")
    return dict(name=name, r2_train=r2_tr, r2_test=r2_te, gap=r2_tr-r2_te, rmse_log=rmse_log, mae_dollars=mae_d, rmse_dollars=rmse_d)

print("\n=== Final tuned comparison (held-out test set) ===")
rf_res = evaluate("Tuned RandomForest", rf_final)
hgb_res = evaluate("Tuned HistGB", hgb_search.best_estimator_)

final_results = pd.DataFrame([rf_res, hgb_res])
final_results.to_csv('/home/claude/airbnb-price-prediction/assets/model_comparison_tuned.csv', index=False)

best_name = "HistGradientBoosting" if hgb_res['r2_test'] >= rf_res['r2_test'] else "RandomForest"
best_pipe = hgb_search.best_estimator_ if hgb_res['r2_test'] >= rf_res['r2_test'] else rf_final
print(f"\nSelected final model: {best_name}")

joblib.dump(best_pipe, '/home/claude/airbnb-price-prediction/app/airbnb_price_pipeline.pkl')
joblib.dump({'feature_cols': feature_cols, 'best_model': best_name,
             'best_params': (hgb_search.best_params_ if best_name=='HistGradientBoosting' else best_rf_params)},
            '/home/claude/airbnb-price-prediction/app/model_meta.pkl')
print("Saved final pipeline to app/airbnb_price_pipeline.pkl")

X_test.assign(price=y_test_d.values, log_price=y_test.values).to_csv('/home/claude/airbnb-price-prediction/data/test_set.csv', index=False)
print("DONE")
