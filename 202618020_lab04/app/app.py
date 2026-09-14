"""
DS605 Lab 4 - Airbnb Price Prediction
Streamlit app that loads the trained pipeline and predicts nightly price
for a new NYC Airbnb listing.

Run locally:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    layout="centered",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MANHATTAN_CENTER = (40.7580, -73.9855)  # Times Square, used as "centrality" reference


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    pipe = joblib.load(os.path.join(APP_DIR, "airbnb_price_pipeline.pkl"))
    meta = joblib.load(os.path.join(APP_DIR, "model_meta.pkl"))
    return pipe, meta


@st.cache_data
def load_neighbourhood_lookup():
    return pd.read_csv(os.path.join(APP_DIR, "neighbourhood_lookup.csv"))


pipe, meta = load_pipeline()
lookup = load_neighbourhood_lookup()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("NYC Airbnb Price Predictor")
st.caption(
    "Estimate a fair nightly price for a New York City Airbnb listing, "
    "based on a model trained on the 2019 Kaggle *NYC Airbnb Open Data* dataset."
)

with st.expander(" About this model"):
    st.markdown(
        f"""
- **Model:** `{meta['best_model']}` (chosen over Linear/Ridge/RandomForest/GradientBoosting
  after comparison and hyperparameter tuning — see the project notebook).
- **Test performance:** R² ≈ 0.65, MAE ≈ \\$42, RMSE ≈ \\$73 (on 2019 NYC data).
- **Limitations:** trained only on 2019 NYC listings with no photos/amenities/description data,
  so treat predictions as a ballpark estimate, not an exact valuation. See the README for details.
        """
    )

st.divider()

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.subheader("Listing details")

col1, col2 = st.columns(2)

with col1:
    borough = st.selectbox("Borough", sorted(lookup["neighbourhood_group"].unique()), index=2)

    neighbourhoods_in_borough = sorted(
        lookup.loc[lookup["neighbourhood_group"] == borough, "neighbourhood"].unique()
    )
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods_in_borough)

    room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])

    minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=3, step=1)

with col2:
    # Auto-fill lat/lon from the chosen neighbourhood, but let the user fine-tune
    match = lookup[
        (lookup["neighbourhood_group"] == borough) & (lookup["neighbourhood"] == neighbourhood)
    ]
    default_lat = float(match["lat"].iloc[0]) if len(match) else 40.73
    default_lon = float(match["lon"].iloc[0]) if len(match) else -73.99

    latitude = st.number_input("Latitude", min_value=40.49, max_value=40.92, value=round(default_lat, 5), format="%.5f")
    longitude = st.number_input("Longitude", min_value=-74.25, max_value=-73.70, value=round(default_lon, 5), format="%.5f")

    number_of_reviews = st.number_input("Number of reviews", min_value=0, max_value=1000, value=5, step=1)
    reviews_per_month = st.number_input("Reviews per month", min_value=0.0, max_value=30.0, value=0.5, step=0.1, format="%.2f")

st.subheader("Host & availability")
col3, col4 = st.columns(2)
with col3:
    calculated_host_listings_count = st.number_input(
        "Host's total listings count", min_value=1, max_value=327, value=1, step=1
    )
with col4:
    availability_365 = st.slider("Availability (days/year)", min_value=0, max_value=365, value=180)

st.divider()
predict_clicked = st.button("🔮 Predict nightly price", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Feature engineering (must mirror the notebook exactly)
# ----------------------------------------------------------------------
def build_feature_row():
    dist_from_center = float(
        np.sqrt((latitude - MANHATTAN_CENTER[0]) ** 2 + (longitude - MANHATTAN_CENTER[1]) ** 2)
    )
    row = {
        "neighbourhood_group": borough,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "latitude": latitude,
        "longitude": longitude,
        "dist_from_center": dist_from_center,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "availability_ratio": availability_365 / 365.0,
        "has_reviews": int(number_of_reviews > 0),
        "never_reviewed": int(number_of_reviews == 0),
        "is_multi_listing_host": int(calculated_host_listings_count > 1),
    }
    return pd.DataFrame([row])[meta["feature_cols"]]


# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------
if predict_clicked:
    X_new = build_feature_row()
    log_pred = pipe.predict(X_new)[0]
    price_pred = float(np.clip(np.expm1(log_pred), 0, None))

    st.success(f"### Estimated nightly price: **${price_pred:,.0f}**")

    # simple uncertainty band based on test-set RMSE ($73), for user context — not a formal CI
    rmse_dollars = 73
    lo, hi = max(0, price_pred - rmse_dollars), price_pred + rmse_dollars
    st.caption(f"Typical model error is about ±\\${rmse_dollars} — realistic range roughly **${lo:,.0f} – ${hi:,.0f}**.")

    with st.expander("See the exact input sent to the model"):
        st.dataframe(X_new.T.rename(columns={0: "value"}), use_container_width=True)
else:
    st.info("Fill in the listing details above and click **Predict nightly price**.")

st.divider()
st.caption("DS605 Fundamentals of Machine Learning — Lab Assignment 4 | Model trained on Kaggle AB_NYC_2019 dataset.")
