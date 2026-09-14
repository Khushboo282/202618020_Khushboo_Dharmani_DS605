"""Task 1 (cont.): Cleaning, transformation, feature engineering"""
import pandas as pd, numpy as np

df = pd.read_csv('/home/claude/airbnb-price-prediction/data/AB_NYC_2019.csv')
print("Original shape:", df.shape)

# 1) Remove invalid price rows (price must be > 0 to be a real listing)
df = df[df['price'] > 0].copy()
print("After removing price<=0:", df.shape)

# 2) Cap extreme outliers instead of dropping (keeps information, avoids distortion)
price_cap = df['price'].quantile(0.99)   # ~$799
min_nights_cap = df['minimum_nights'].quantile(0.99)  # ~45
print(f"price_cap={price_cap}, min_nights_cap={min_nights_cap}")
df = df[df['price'] <= price_cap].copy()
df['minimum_nights'] = df['minimum_nights'].clip(upper=min_nights_cap)

# 3) Missing values
# reviews_per_month is NaN exactly when a listing has never been reviewed -> fill 0 (meaningful, not random)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
# last_review missing -> flag as "never reviewed"; drop raw date (not usable directly by model)
df['never_reviewed'] = df['last_review'].isna().astype(int)
df = df.drop(columns=['last_review'])
# name/host_name missing -> irrelevant free text / IDs, drop
df = df.drop(columns=['name', 'host_name', 'id', 'host_id'])

print("\nMissing after cleaning:\n", df.isnull().sum().sum(), "total nulls")

# 4) Feature engineering
# Distance from Manhattan center (Times Square) as a proxy for "how central" a listing is
manhattan_center = (40.7580, -73.9855)
df['dist_from_center'] = np.sqrt((df['latitude']-manhattan_center[0])**2 + (df['longitude']-manhattan_center[1])**2)

# Review activity bucket
df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)

# Availability ratio (0-1)
df['availability_ratio'] = df['availability_365'] / 365.0

# Host scale: is this a multi-listing ("business") host?
df['is_multi_listing_host'] = (df['calculated_host_listings_count'] > 1).astype(int)

# Log-transform skewed numeric features (log1p handles zeros)
for col in ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']:
    df[f'log_{col}'] = np.log1p(df[col])

# Target: log-transform price for modeling (train models on log scale, invert with expm1 for real-world $ output)
df['log_price'] = np.log1p(df['price'])

# 5) Neighbourhood: 221 unique values -> too high cardinality for one-hot.
# Reduce by target-encoding-free approach: keep neighbourhood_group (5 values, one-hot)
# and use neighbourhood only as a frequency/mean-encoded feature computed inside the pipeline (to avoid leakage, done after split).

print("\nFinal columns:", list(df.columns))
print("Final shape:", df.shape)
df.to_csv('/home/claude/airbnb-price-prediction/data/airbnb_cleaned.csv', index=False)
print("\nSaved cleaned dataset.")
