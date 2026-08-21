
# DS605: Fundamentals of Machine Learning — Lab Assignment 3

**Title:** Scikit-learn: Data Preprocessing and Model Performance Evaluation
**Name:** Khushboo Dharmani
**ID:** _[202618020]_
**Dataset:** [Kaggle Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (`hotel_bookings.csv`)

## Preprocessing Choices
- **Target:** `is_canceled` (binary: 0 = not cancelled, 1 = cancelled). Class distribution is roughly 63% / 37%, so `stratify=y` was used on the split and precision/recall/F1 are reported alongside accuracy.
- **Dropped `company`** (~94% missing) — too sparse to impute meaningfully.
- **`agent`** (~14% missing) — missing values filled with `0`, treated as "no agent used" (direct booking) rather than dropped.
- **`country`** and **`children`** — low missingness, imputed within the pipeline (categorical / numerical respectively).
- **Dropped `reservation_status` and `reservation_status_date`** — these leak the final outcome directly.
- **Outliers:** removed only clear/extreme cases using IQR with a conservative multiplier (`k=3`) on `adr`, `lead_time`, and invalid `adults` counts (0 or >10). Full dataset went from 119,390 → 118,483 rows.
- **Split:** `train_test_split(test_size=0.2, stratify=y, random_state=42)`, reused for all four experiments.
- **Pipeline A:** `KNNImputer(n_neighbors=5)` + `StandardScaler` (numerical), `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")` (categorical).
- **Pipeline B:** same as A but with `MinMaxScaler` instead of `StandardScaler`.
- Both pipelines built with `ColumnTransformer` + `Pipeline`, fitted only on the training split.

## Final Comparison Table

| Model | Train Acc. | Test Acc. | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Logistic Regression + Pipeline A (StandardScaler) | 0.8185 | 0.8180 | 0.8063 | 0.6689 | 0.7312 |
| Logistic Regression + Pipeline B (MinMaxScaler)    | 0.8147 | 0.8146 | 0.8047 | 0.6588 | 0.7245 |
| Decision Tree + Pipeline A (StandardScaler)        | 0.9961 | 0.8642 | 0.8120 | 0.8234 | 0.8177 |
| Decision Tree + Pipeline B (MinMaxScaler)           | 0.9961 | 0.8643 | 0.8123 | 0.8235 | 0.8179 |

## Final Observations
1. **Best overall combination:** Decision Tree + Pipeline B (MinMaxScaler), marginally ahead of Decision Tree + Pipeline A, both well ahead of either Logistic Regression run on test accuracy, recall, and F1.
2. **Scaler effect on Logistic Regression:** StandardScaler slightly outperforms MinMaxScaler (81.8% vs 81.5% test accuracy), consistent with Logistic Regression's gradient-based optimizer converging better on zero-centered, unit-variance features.
3. **Scaler effect on Decision Tree:** Negligible — test accuracy differs by only ~0.01 percentage points between A and B, because trees split on thresholds/order, not on feature magnitude.
4. **Overfitting:** The Decision Tree overfits noticeably (train accuracy ~99.6% vs test ~86.4%, a ~13-point gap), while Logistic Regression shows almost no train-test gap. Pruning (`max_depth`, `min_samples_leaf`) would likely reduce Decision Tree overfitting.
5. **Precision/recall tradeoff:** Decision Tree has meaningfully higher recall (~0.82) than Logistic Regression (~0.66-0.67) at similar precision (~0.81), so it catches more true cancellations — useful if the goal is proactively flagging at-risk bookings.




