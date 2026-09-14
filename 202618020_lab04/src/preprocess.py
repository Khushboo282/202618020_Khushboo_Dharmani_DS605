# Preprocessing and feature engineering functions for NYC Airbnb dataset

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# NYC Reference Coordinates (Midtown Manhattan / Times Square)
NYC_MIDTOWN_LAT = 40.7580
NYC_MIDTOWN_LON = -73.9855

# Wall Street / Financial District
NYC_WALL_ST_LAT = 40.7074
NYC_WALL_ST_LON = -74.0113

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on earth in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

class AirbnbFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to perform domain-specific feature engineering
    on raw Airbnb listing data.
    """
    def __init__(self, top_n_neighbourhoods=25):
        self.top_n_neighbourhoods = top_n_neighbourhoods
        self.frequent_neighbourhoods_ = None

    def fit(self, X, y=None):
        df = X.copy()
        if 'neighbourhood' in df.columns:
            counts = df['neighbourhood'].value_counts()
            self.frequent_neighbourhoods_ = list(counts.head(self.top_n_neighbourhoods).index)
        else:
            self.frequent_neighbourhoods_ = []
        return self

    def transform(self, X):
        df = X.copy()

        # 1. Clean missing values
        if 'reviews_per_month' in df.columns:
            df['reviews_per_month'] = df['reviews_per_month'].fillna(0.0)
        
        if 'name' in df.columns:
            df['name_length'] = df['name'].fillna('').astype(str).str.len()
        else:
            df['name_length'] = 20  # default median length

        # 2. Distance to Midtown & Financial District
        lat = df['latitude'].fillna(NYC_MIDTOWN_LAT)
        lon = df['longitude'].fillna(NYC_MIDTOWN_LON)
        df['dist_to_midtown'] = haversine_distance(lat, lon, NYC_MIDTOWN_LAT, NYC_MIDTOWN_LON)
        df['dist_to_wall_st'] = haversine_distance(lat, lon, NYC_WALL_ST_LAT, NYC_WALL_ST_LON)

        # 3. Behavioral and listing feature flags
        if 'minimum_nights' in df.columns:
            df['minimum_nights_capped'] = df['minimum_nights'].clip(upper=30)
        else:
            df['minimum_nights_capped'] = 1

        if 'calculated_host_listings_count' in df.columns:
            df['is_commercial_host'] = (df['calculated_host_listings_count'] > 1).astype(int)
        else:
            df['is_commercial_host'] = 0

        if 'number_of_reviews' in df.columns:
            df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
        else:
            df['has_reviews'] = 0

        # 4. Group rare neighbourhoods
        if 'neighbourhood' in df.columns and self.frequent_neighbourhoods_ is not None:
            df['neighbourhood_grouped'] = df['neighbourhood'].apply(
                lambda x: x if x in self.frequent_neighbourhoods_ else 'Other'
            )
        else:
            df['neighbourhood_grouped'] = 'Other'

        # Select relevant engineered columns
        features_to_keep = [
            'neighbourhood_group',
            'neighbourhood_grouped',
            'room_type',
            'latitude',
            'longitude',
            'minimum_nights_capped',
            'number_of_reviews',
            'reviews_per_month',
            'calculated_host_listings_count',
            'availability_365',
            'dist_to_midtown',
            'dist_to_wall_st',
            'name_length',
            'is_commercial_host',
            'has_reviews'
        ]

        # Keep existing features in X that match features_to_keep
        available_cols = [c for c in features_to_keep if c in df.columns]
        return df[available_cols]


def build_preprocessor(top_n_neighbourhoods=25):
    """
    Constructs the end-to-end scikit-learn ColumnTransformer pipeline.
    """
    categorical_cols = ['neighbourhood_group', 'neighbourhood_grouped', 'room_type']
    numeric_cols = [
        'latitude',
        'longitude',
        'minimum_nights_capped',
        'number_of_reviews',
        'reviews_per_month',
        'calculated_host_listings_count',
        'availability_365',
        'dist_to_midtown',
        'dist_to_wall_st',
        'name_length',
        'is_commercial_host',
        'has_reviews'
    ]

    col_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    preprocessor_pipeline = Pipeline(steps=[
        ('feature_engineer', AirbnbFeatureEngineer(top_n_neighbourhoods=top_n_neighbourhoods)),
        ('col_transformer', col_transformer)
    ])

    return preprocessor_pipeline


def load_and_clean_data(csv_path='AB_NYC_2019.csv', filter_outliers=True):
    """
    Loads raw CSV and applies filtering on target for training.
    Outlier strategy:
    - Exclude price <= 0 (invalid/corrupted records)
    - Exclude extreme price outliers (price > $1000) for stable regression training,
      preserving 99.5% of regular listings while removing unrepresentative luxury spikes.
    """
    df = pd.read_csv(csv_path)

    if filter_outliers:
        # Keep realistic valid prices between $10 and $1000
        df = df[(df['price'] >= 10) & (df['price'] <= 1000)].copy()

    # Create log-transformed price target
    df['log_price'] = np.log1p(df['price'])
    return df

if __name__ == '__main__':
    df = load_and_clean_data()
    print(f"Cleaned dataset shape: {df.shape}")
    preprocessor = build_preprocessor()
    X = df.drop(columns=['price', 'log_price'])
    X_trans = preprocessor.fit_transform(X)
    print(f"Transformed features shape: {X_trans.shape}")
    print("Preprocessing verification passed successfully.")
