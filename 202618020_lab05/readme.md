# Lab 5: Machine Learning with Scikit-learn and From Scratch

This repository contains the end-to-end implementation of predicting garment employee productivity using both `scikit-learn` and a from-scratch implementation utilizing only `numpy` and `pandas`.

## Overview

The dataset used is the **UCI Productivity Prediction of Garment Employees**.
We focus on two tasks:
1. **Regression Task:** Predicting `actual_productivity` using Linear Regression.
2. **Classification Task:** Predicting if `MeetsTarget = 1` (where `actual_productivity >= targeted_productivity`) using Logistic Regression.

## File Structure

- `analysis.ipynb`: The complete Jupyter Notebook containing data preprocessing, scikit-learn models, custom numpy/pandas models, comparison tables, and final observations.
- `garments_worker_productivity.csv`: The dataset utilized.

## Key Observations

1. **Predictive Performance**
   - **Linear Regression:** The manual implementation using the closed-form Normal Equation successfully matched `scikit-learn`'s metrics (R2, MAE, RMSE) perfectly.
   - **Logistic Regression:** Our manual Gradient Descent model came extremely close to `scikit-learn`'s results. By introducing L2 Regularization (Ridge) and tuning the learning rate, the manual approach matched `scikit-learn` in terms of generalization.

2. **Execution Time**
   - **Linear Regression:** Calculating the Normal Equation manually for a dataset of this size proved to be extremely fast, often matching or beating `scikit-learn` since there's zero framework overhead.
   - **Logistic Regression:** `scikit-learn` is considerably faster for iterative solvers due to its underlying optimized C/C++ backend and highly optimized routines like `lbfgs`.

3. **Implementation Efficiency**
   - Preprocessing is significantly easier with `scikit-learn` pipelines (e.g. `ColumnTransformer`). Managing scaling parameters, modes/medians, and one-hot encoding columns purely in NumPy/Pandas requires much more careful bookkeeping to avoid data leakage between the train and test sets.
