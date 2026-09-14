# NYC Airbnb Price Prediction — End-to-End ML Project

**Course:** DS605 – Fundamentals of Machine Learning · Lab Assignment 4
**Dataset:** [Kaggle — New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (`AB_NYC_2019.csv`)

An end-to-end machine learning workflow — data cleaning, feature engineering, model comparison,
hyperparameter tuning, and a deployable Streamlit app — that predicts the nightly price of a
NYC Airbnb listing.

**Deployed app link:** _add after deploying_

---

## 📁 Repository structure

```
airbnb-price-prediction/
├── data/
│   ├── AB_NYC_2019.csv           # raw Kaggle dataset
│   ├── airbnb_cleaned.csv        # cleaned + feature-engineered dataset
│   └── test_set.csv              # held-out test split
├── notebooks/
│   └── airbnb_price_prediction.ipynb   # full EDA -> cleaning -> modeling -> evaluation
├── app/
│   ├── app.py                          # Streamlit application
│   ├── airbnb_price_pipeline.pkl       # saved sklearn Pipeline (preprocessing + model)
│   ├── model_meta.pkl                  # feature list + model metadata
│   └── neighbourhood_lookup.csv        # borough/neighbourhood -> avg lat/lon (used by the app)
├── assets/                             # EDA plots, model comparison plots, screenshots
├── scripts/                            # standalone .py versions of the notebook steps
├── requirements.txt
└── README.md
```

---

## Task 1 — Data Analysis & Preparation

- **48,895** raw listings, 16 columns.
- **Cleaning:** dropped 11 listings with `price = 0` (invalid); capped `price` and `minimum_nights`
  at their 99th percentiles (~\$799 and ~45 nights) to limit the influence of extreme outliers
  without discarding the whole distribution tail.
- **Missing values:** `reviews_per_month` / `last_review` were missing together for ~20.6% of rows —
  this simply means "never reviewed," so `reviews_per_month` was filled with 0 and a `never_reviewed`
  flag was added instead of dropping rows. `name`, `host_name`, `id`, `host_id` were dropped
  (free text / identifiers, not predictive).
- **Transformations:** `price` (target) and skewed count features (`minimum_nights`,
  `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`) were log-transformed
  (`log1p`) to reduce skew; models are trained on `log1p(price)` and inverted with `expm1`.
- **Feature engineering:** `dist_from_center` (distance from Times Square, proxy for centrality),
  `availability_ratio`, `has_reviews`, `is_multi_listing_host` (business vs. individual host).
- **High-cardinality `neighbourhood`** (221 unique values): handled with an in-pipeline, cross-fitted
  `sklearn.preprocessing.TargetEncoder` instead of one-hot encoding (avoids a 221-column sparse blow-up
  and target leakage). `neighbourhood_group` (5 boroughs) and `room_type` (3 values) are one-hot encoded.

Final cleaned dataset: **48,410 rows × 21 columns**, zero missing values.

**Key patterns found:** `room_type` and borough are the dominant visible drivers of price
(Entire home/apt & Manhattan command the highest prices); price has a strong geographic gradient
centered on Manhattan; raw linear correlations with price are weak (|r| < 0.16), indicating the
relationship is largely non-linear/categorical rather than linear.

## Task 2 — Model Training & Evaluation

Five regression models were trained and compared on an 80/20 train/test split (`random_state=42`):

| Model | Test R² | Train R² | Train/Test Gap | MAE ($) | RMSE ($) |
|---|---|---|---|---|---|
| Linear Regression | 0.596 | 0.592 | -0.004 | 45.83 | 79.79 |
| Ridge | 0.596 | 0.592 | -0.004 | 45.83 | 79.80 |
| Random Forest (baseline) | 0.650 | 0.790 | 0.140 | 41.90 | 73.02 |
| Gradient Boosting | 0.637 | 0.643 | 0.006 | 43.30 | 76.23 |
| Hist Gradient Boosting (baseline) | 0.649 | 0.686 | 0.037 | 42.32 | 73.79 |

Linear models **underfit** (R² plateaus at ~0.60) since price depends on non-linear
location × room-type interactions. **Random Forest** and **HistGradientBoosting** were carried
forward for tuning.

**Hyperparameter tuning** *(single-core sandbox, so RandomForest — slow per fit — was tuned with a
light manual grid on a validation split, while HistGB — fast per fit — got a full 5-fold
`RandomizedSearchCV`)*:

| Model (tuned) | Test R² | Train R² | Gap | MAE ($) | RMSE ($) |
|---|---|---|---|---|---|
| Random Forest | 0.6505 | 0.7817 | **0.1312** (overfitting) | 41.88 | 72.98 |
| **Hist Gradient Boosting (final)** | **0.6509** | 0.6960 | **0.0451** | **42.12** | **73.41** |

**Final model: `HistGradientBoostingRegressor`** — essentially tied on test R² with the tuned
RandomForest, but with a far smaller overfitting gap, making it the more reliable generalizer.
Final hyperparameters: `max_iter=450, max_depth=8, learning_rate=0.05, min_samples_leaf=20, l2_regularization=0.0`.

**Top features by permutation importance:** `room_type` (dominant, 0.65), `dist_from_center`,
`availability_365`, `minimum_nights`, `neighbourhood`.

The full pipeline (`ColumnTransformer` preprocessing + tuned `HistGradientBoostingRegressor`) is
saved as `app/airbnb_price_pipeline.pkl` via `joblib`, so the exact same transformations are applied
to new inputs at inference time.

## Task 3 — Streamlit Application

`app/app.py` loads the saved pipeline and provides a form for: borough, neighbourhood
(auto-filters + auto-fills coordinates), room type, minimum nights, review activity, host listing
count, and availability. It returns an estimated nightly price plus a rough ± error band based on
test-set RMSE.



**Deployed app link:** _add after deploying_

## Task 4 — Final Summary

See the notebook's final section (`notebooks/airbnb_price_prediction.ipynb`) for the full write-up.
In short: the model explains ~65% of price variance using only structural listing attributes
(location, room type, availability, review activity). The remaining ~35% is driven by factors this
2019 tabular dataset doesn't capture — photos, amenities, listing quality, host reputation/text
description, and real-time demand/seasonality. Predictions should be read as a **ballpark estimate**,
not a precise valuation, and the model reflects **2019 NYC pricing**, so absolute dollar values will
be stale for current markets or other cities without retraining.

---

