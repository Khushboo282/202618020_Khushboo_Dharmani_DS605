# DS605: Fundamentals of Machine Learning — Lab Assignment 2

**Title:** Vectorized Programming with NumPy and Data Wrangling with Pandas
**Name:** Khushboo Dharmani
**ID:** 202618020
**Dataset:** [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data) (`train.csv`)

## Project Details

This repository contains a complete solution to Lab 2, covering:

**Part A — NumPy**
- Task 1: Array creation, indexing, statistics
- Task 2: Vectorized arithmetic & linear algebra (matrix mult, transpose, determinant, inverse)
- Task 3: Normal distribution sampling and histogram

**Part B — Pandas (Titanic dataset)**
- Task 4: Load & inspect data (`head`, `tail`, `info`, `describe`, `loc` vs `iloc`)
- Task 5: Boolean filtering / `query()` questions
- Task 6: `groupby()` aggregations (survival rate by sex, class, embarkation, etc.)
- Task 7: Missing value analysis, multiple imputation strategies, Fare outlier detection (IQR)
- Task 8: Feature engineering (`FamilySize`, `IsAlone`) and pivot table
- Task 9: Correlation heatmap, survival-rate bar chart, Age vs Fare scatter, written observations

## Key Observations

1. Sex is the strongest predictor of survival — women survived at a much higher rate than men.
2. Passenger class matters — 1st class passengers survived at a higher rate than 3rd class.
3. Fare correlates positively with survival, largely as a proxy for Pclass.
4. Age has only a weak negative correlation with survival overall.
5. Passengers traveling completely alone, and those in very large families, survived at lower
   rates than passengers in small families (2–4 members).
6. Cabin and Age have the most missing values; Cabin is missing for most passengers.
7. The highest-survival group is 1st-class women; the lowest-survival group is 3rd-class men.

