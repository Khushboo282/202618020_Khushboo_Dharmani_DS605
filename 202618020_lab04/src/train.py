# Model training, comparison, and hyperparameter tuning for Airbnb price prediction

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb

from preprocess import load_and_clean_data, build_preprocessor

FIGURES_DIR = os.path.join('reports', 'figures')
MODELS_DIR = 'models'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def evaluate_predictions(y_true_log, y_pred_log):
    """
    Computes regression evaluation metrics both in log-space and original dollar space.
    """
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)

    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2_orig = r2_score(y_true_orig, y_pred_orig)
    r2_log = r2_score(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))

    return {
        'MAE ($)': round(float(mae_orig), 2),
        'RMSE ($)': round(float(rmse_orig), 2),
        'R2 (Price)': round(float(r2_orig), 4),
        'R2 (Log)': round(float(r2_log), 4),
        'RMSE (Log)': round(float(rmse_log), 4)
    }

def train_and_evaluate_all():
    print("==================================================")
    print("DS605 Task 2: Model Training & Evaluation Benchmark")
    print("==================================================")

    # 1. Load data
    df = load_and_clean_data(filter_outliers=True)
    feature_cols = [c for c in df.columns if c not in ['price', 'log_price', 'id', 'host_name']]
    X = df[feature_cols]
    y = df['log_price']

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

    # Fit preprocessor on X_train only to prevent data leakage
    preprocessor = build_preprocessor(top_n_neighbourhoods=25)
    print("Fitting preprocessor pipeline on training set...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    print(f"Transformed feature dimension: {X_train_trans.shape[1]}")

    # Define candidate regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),
        'Lasso Regression': Lasso(alpha=0.001, max_iter=2000),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=15, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, n_jobs=-1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, verbosity=0)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_t = time.time()
        model.fit(X_train_trans, y_train)
        elapsed = time.time() - start_t

        # Predict train & test
        pred_train_log = model.predict(X_train_trans)
        pred_test_log = model.predict(X_test_trans)

        train_metrics = evaluate_predictions(y_train, pred_train_log)
        test_metrics = evaluate_predictions(y_test, pred_test_log)

        # Cross-validation score on train set (3-fold for speed during screening)
        cv_scores = cross_val_score(model, X_train_trans, y_train, cv=3, scoring='r2')

        results.append({
            'Model': name,
            'Train R2 (Log)': train_metrics['R2 (Log)'],
            'Test R2 (Log)': test_metrics['R2 (Log)'],
            'Train MAE ($)': train_metrics['MAE ($)'],
            'Test MAE ($)': test_metrics['MAE ($)'],
            'Train RMSE ($)': train_metrics['RMSE ($)'],
            'Test RMSE ($)': test_metrics['RMSE ($)'],
            'Test R2 (Price)': test_metrics['R2 (Price)'],
            'CV R2 Mean': round(float(cv_scores.mean()), 4),
            'Fit Time (s)': round(elapsed, 2)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values('Test R2 (Log)', ascending=False)
    print("\n--- Model Benchmark Results ---")
    print(results_df.to_string(index=False))

    # Save benchmark table
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_benchmark.csv'), index=False)

    # Plot Model Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(results_df))
    width = 0.35

    plt.bar(x_pos - width/2, results_df['Train R2 (Log)'], width, label='Train $R^2$ (Log)', color='#3498db')
    plt.bar(x_pos + width/2, results_df['Test R2 (Log)'], width, label='Test $R^2$ (Log)', color='#2ecc71')

    plt.xticks(x_pos, results_df['Model'], rotation=25, ha='right', fontsize=10)
    plt.ylabel('$R^2$ Score', fontsize=12)
    plt.title('Regression Models Comparison: Train vs Test $R^2$ (Log-Space)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    comparison_fig = os.path.join(FIGURES_DIR, 'model_comparison.png')
    plt.savefig(comparison_fig, dpi=300)
    plt.close()
    print(f"Saved: {comparison_fig}")

    # 3. Hyperparameter Tuning on the Best Model (LightGBM)
    print("\n==================================================")
    print("Hyperparameter Tuning for LightGBM Regressor")
    print("==================================================")
    param_grid = {
        'n_estimators': [150, 250],
        'learning_rate': [0.05, 0.08],
        'num_leaves': [31, 45],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    base_lgb = lgb.LGBMRegressor(random_state=42, verbose=-1)
    grid_search = GridSearchCV(
        estimator=base_lgb,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_trans, y_train)
    best_lgb = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Evaluate Tuned Model
    tuned_train_pred = best_lgb.predict(X_train_trans)
    tuned_test_pred = best_lgb.predict(X_test_trans)
    tuned_train_metrics = evaluate_predictions(y_train, tuned_train_pred)
    tuned_test_metrics = evaluate_predictions(y_test, tuned_test_pred)

    print("\nTuned LightGBM Performance:")
    print(f"Train R2 (Log): {tuned_train_metrics['R2 (Log)']:.4f} | Test R2 (Log): {tuned_test_metrics['R2 (Log)']:.4f}")
    print(f"Train MAE: ${tuned_train_metrics['MAE ($)']:.2f} | Test MAE: ${tuned_test_metrics['MAE ($)']:.2f}")
    print(f"Train RMSE: ${tuned_train_metrics['RMSE ($)']:.2f} | Test RMSE: ${tuned_test_metrics['RMSE ($)']:.2f}")
    print(f"Test R2 (Orig Price): {tuned_test_metrics['R2 (Price)']:.4f}")

    # 4. Overfitting / Underfitting Diagnostics Plots
    # Residuals & Actual vs Predicted
    y_test_orig = np.expm1(y_test)
    tuned_pred_orig = np.clip(np.expm1(tuned_test_pred), a_min=0, a_max=None)
    residuals = y_test_orig - tuned_pred_orig

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Actual vs Predicted Plot
    axes[0].scatter(y_test_orig, tuned_pred_orig, alpha=0.25, color='#2980b9', s=15)
    axes[0].plot([0, 1000], [0, 1000], 'r--', lw=2, label='Ideal Fit (y = x)')
    axes[0].set_title('Actual vs Predicted Nightly Price ($)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Actual Price ($)', fontsize=11)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=11)
    axes[0].set_xlim(0, 1000)
    axes[0].set_ylim(0, 1000)
    axes[0].legend()

    # Residuals Distribution Plot
    sns.histplot(residuals[(residuals >= -300) & (residuals <= 300)], kde=True, ax=axes[1], color='#8e44ad', bins=50)
    axes[1].axvline(0, color='r', linestyle='--', lw=2)
    axes[1].set_title('Residuals Distribution (Actual - Predicted)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Residual ($)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)

    plt.tight_layout()
    pred_res_fig = os.path.join(FIGURES_DIR, 'actual_vs_predicted_and_residuals.png')
    plt.savefig(pred_res_fig, dpi=300)
    plt.close()
    print(f"Saved: {pred_res_fig}")

    # Feature Importance Plot
    # Get feature names from ColumnTransformer
    col_trans = preprocessor.named_steps['col_transformer']
    numeric_names = [
        'latitude', 'longitude', 'minimum_nights_capped', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
        'dist_to_midtown', 'dist_to_wall_st', 'name_length', 'is_commercial_host', 'has_reviews'
    ]
    cat_names = list(col_trans.named_transformers_['cat'].get_feature_names_out(
        ['neighbourhood_group', 'neighbourhood_grouped', 'room_type']
    ))
    all_feature_names = numeric_names + cat_names

    importances = best_lgb.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Top 15 Feature Importances (Tuned LightGBM Regressor)', fontsize=13, fontweight='bold')
    plt.xlabel('Importance (Split Count)', fontsize=11)
    plt.tight_layout()
    feat_fig = os.path.join(FIGURES_DIR, 'feature_importance.png')
    plt.savefig(feat_fig, dpi=300)
    plt.close()
    print(f"Saved: {feat_fig}")

    # 5. Save End-to-End Pipeline
    print("\n==================================================")
    print("Building & Persisting End-to-End Pipeline")
    print("==================================================")
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_lgb)
    ])

    pipeline_path = os.path.join(MODELS_DIR, 'pipeline.joblib')
    joblib.dump(full_pipeline, pipeline_path)
    print(f"Saved full pipeline to: {pipeline_path}")

    # Export metrics json for app / report reference
    metrics_summary = {
        'best_model': 'LightGBM Regressor',
        'best_params': grid_search.best_params_,
        'train_metrics': tuned_train_metrics,
        'test_metrics': tuned_test_metrics,
        'feature_count': len(all_feature_names),
        'top_features': feat_imp_df.to_dict(orient='records')
    }
    metrics_json_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved metrics summary to: {metrics_json_path}")

    # Verify reload and sample inference
    print("\nVerifying pipeline reload and sample inference...")
    loaded_pipe = joblib.load(pipeline_path)
    sample_input = X_test.head(3)
    sample_pred_log = loaded_pipe.predict(sample_input)
    sample_pred_orig = np.expm1(sample_pred_log)
    print(f"Sample predictions (dollars): {sample_pred_orig.round(2)}")
    print("Pipeline validation successful!")

if __name__ == '__main__':
    train_and_evaluate_all()
