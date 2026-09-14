# Quick test script to verify model predictions on sample listings
import sys
import os
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('src'))
from preprocess import AirbnbFeatureEngineer

def test_inference():
    print("Loading saved pipeline...")
    pipe = joblib.load('models/pipeline.joblib')

    scenarios = pd.DataFrame([
        {
            'name': 'Midtown Manhattan Entire Apt',
            'neighbourhood_group': 'Manhattan',
            'neighbourhood': 'Midtown',
            'room_type': 'Entire home/apt',
            'latitude': 40.7580,
            'longitude': -73.9855,
            'minimum_nights': 3,
            'number_of_reviews': 40,
            'reviews_per_month': 2.0,
            'calculated_host_listings_count': 1,
            'availability_365': 200
        },
        {
            'name': 'Williamsburg Brooklyn Private Room',
            'neighbourhood_group': 'Brooklyn',
            'neighbourhood': 'Williamsburg',
            'room_type': 'Private room',
            'latitude': 40.7083,
            'longitude': -73.9535,
            'minimum_nights': 2,
            'number_of_reviews': 60,
            'reviews_per_month': 2.5,
            'calculated_host_listings_count': 1,
            'availability_365': 90
        },
        {
            'name': 'Astoria Queens Budget Stay',
            'neighbourhood_group': 'Queens',
            'neighbourhood': 'Astoria',
            'room_type': 'Private room',
            'latitude': 40.7644,
            'longitude': -73.9235,
            'minimum_nights': 1,
            'number_of_reviews': 10,
            'reviews_per_month': 0.8,
            'calculated_host_listings_count': 1,
            'availability_365': 300
        },
        {
            'name': 'Mott Haven Bronx Shared Room',
            'neighbourhood_group': 'Bronx',
            'neighbourhood': 'Mott Haven',
            'room_type': 'Shared room',
            'latitude': 40.8080,
            'longitude': -73.9200,
            'minimum_nights': 1,
            'number_of_reviews': 5,
            'reviews_per_month': 0.5,
            'calculated_host_listings_count': 2,
            'availability_365': 350
        }
    ])

    pred_log = pipe.predict(scenarios)
    pred_dollars = np.expm1(pred_log)

    print("\n--- Realistic Input Test Predictions ---")
    for idx, row in scenarios.iterrows():
        print(f"Scenario: {row['name']}")
        print(f"  Borough: {row['neighbourhood_group']} | Type: {row['room_type']}")
        print(f"  Predicted Nightly Price: ${pred_dollars[idx]:.2f}\n")

if __name__ == '__main__':
    test_inference()
