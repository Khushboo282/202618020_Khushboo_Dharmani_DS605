"""Task 1 (cont.): Cleaning, transformation, feature engineering"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "AB_NYC_2019.csv")
print("Original shape:", df.shape)

df = df[df["price"] > 0].copy()
print("After removing price<=0:", df.shape)

price_cap = df["price"].quantile(0.99)
min_nights_cap = df["minimum_nights"].quantile(0.99)
print(f"price_cap={price_cap}, min_nights_cap={min_nights_cap}")
df = df[df["price"] <= price_cap].copy()
df["minimum_nights"] = df["minimum_nights"].clip(upper=min_nights_cap)

df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df["never_reviewed"] = df["last_review"].isna().astype(int)
df = df.drop(columns=["last_review", "name", "host_name", "id", "host_id"])

print("\nMissing after cleaning:\n", df.isnull().sum().sum(), "total nulls")

manhattan_center = (40.7580, -73.9855)
df["dist_from_center"] = np.sqrt(
    (df["latitude"] - manhattan_center[0]) ** 2
    + (df["longitude"] - manhattan_center[1]) ** 2
)
df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)
df["availability_ratio"] = df["availability_365"] / 365.0
df["is_multi_listing_host"] = (df["calculated_host_listings_count"] > 1).astype(int)

for col in [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
]:
    df[f"log_{col}"] = np.log1p(df[col])

df["log_price"] = np.log1p(df["price"])

print("\nFinal columns:", list(df.columns))
print("Final shape:", df.shape)
df.to_csv(ROOT / "data" / "airbnb_cleaned.csv", index=False)
print("\nSaved cleaned dataset.")
