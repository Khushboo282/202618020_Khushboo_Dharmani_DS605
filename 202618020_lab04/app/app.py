"""
DS605 Lab 4 - Airbnb Price Prediction
Streamlit app that loads the trained pipeline and predicts nightly price
for a new NYC Airbnb listing.

Run locally (from the lab04 folder):
    streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    layout="centered",
)

APP_DIR = Path(__file__).resolve().parent
MANHATTAN_CENTER = (40.7580, -73.9855)


@st.cache_resource
def load_pipeline():
    pipe_path = APP_DIR / "airbnb_price_pipeline.pkl"
    meta_path = APP_DIR / "model_meta.pkl"
    if not pipe_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Missing model files. From the lab04 folder run: "
            "python scripts/save_final_pipeline.py"
        )
    pipe = joblib.load(pipe_path)
    meta = joblib.load(meta_path)
    return pipe, meta


@st.cache_data
def load_neighbourhood_lookup():
    return pd.read_csv(APP_DIR / "neighbourhood_lookup.csv")


try:
    pipe, meta = load_pipeline()
    lookup = load_neighbourhood_lookup()
    load_error = None
except Exception as exc:
    pipe, meta, lookup = None, None, None
    load_error = str(exc)

st.title("NYC Airbnb Price Predictor")
st.caption(
    "Estimate a fair nightly price for a New York City Airbnb listing, "
    "based on a model trained on the 2019 Kaggle NYC Airbnb Open Data dataset."
)

if load_error:
    st.error(
        "The trained model could not be loaded, so predictions are unavailable."
    )
    st.code(load_error)
    st.stop()

r2 = meta.get("r2_test", 0.65)
mae = meta.get("mae_dollars", 42)
rmse = int(round(meta.get("rmse_dollars", 73)))

with st.expander("About this model"):
    st.markdown(
        f"""
- **Model:** `{meta['best_model']}` (chosen over Linear/Ridge/RandomForest/GradientBoosting
  after comparison and hyperparameter tuning — see the project notebook).
- **Test performance:** R² ≈ {r2:.2f}, MAE ≈ ${mae:.0f}, RMSE ≈ ${rmse} (on 2019 NYC data).
- **Limitations:** trained only on 2019 NYC listings with no photos/amenities/description data,
  so treat predictions as a ballpark estimate, not an exact valuation. See the README for details.
        """
    )

st.divider()
st.subheader("Listing details")

col1, col2 = st.columns(2)
boroughs = sorted(lookup["neighbourhood_group"].unique())
manhattan_index = boroughs.index("Manhattan") if "Manhattan" in boroughs else 0

with col1:
    borough = st.selectbox("Borough", boroughs, index=manhattan_index)
    neighbourhoods_in_borough = sorted(
        lookup.loc[lookup["neighbourhood_group"] == borough, "neighbourhood"].unique()
    )
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods_in_borough)
    room_type = st.selectbox(
        "Room type", ["Entire home/apt", "Private room", "Shared room"]
    )
    minimum_nights = st.number_input(
        "Minimum nights", min_value=1, max_value=365, value=3, step=1
    )

with col2:
    match = lookup[
        (lookup["neighbourhood_group"] == borough)
        & (lookup["neighbourhood"] == neighbourhood)
    ]
    default_lat = float(match["lat"].iloc[0]) if len(match) else 40.73
    default_lon = float(match["lon"].iloc[0]) if len(match) else -73.99

    place_key = f"{borough}|{neighbourhood}"
    if st.session_state.get("_place_key") != place_key:
        st.session_state["_place_key"] = place_key
        st.session_state["latitude"] = round(default_lat, 5)
        st.session_state["longitude"] = round(default_lon, 5)

    latitude = st.number_input(
        "Latitude",
        min_value=40.49,
        max_value=40.92,
        key="latitude",
        format="%.5f",
    )
    longitude = st.number_input(
        "Longitude",
        min_value=-74.25,
        max_value=-73.70,
        key="longitude",
        format="%.5f",
    )
    number_of_reviews = st.number_input(
        "Number of reviews", min_value=0, max_value=1000, value=5, step=1
    )
    reviews_per_month = st.number_input(
        "Reviews per month",
        min_value=0.0,
        max_value=30.0,
        value=0.5,
        step=0.1,
        format="%.2f",
    )

st.subheader("Host & availability")
col3, col4 = st.columns(2)
with col3:
    calculated_host_listings_count = st.number_input(
        "Host's total listings count", min_value=1, max_value=327, value=1, step=1
    )
with col4:
    availability_365 = st.slider(
        "Availability (days/year)", min_value=0, max_value=365, value=180
    )

st.divider()
predict_clicked = st.button(
    "Predict nightly price", type="primary", use_container_width=True
)


def build_feature_row():
    dist_from_center = float(
        np.sqrt(
            (latitude - MANHATTAN_CENTER[0]) ** 2
            + (longitude - MANHATTAN_CENTER[1]) ** 2
        )
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


if predict_clicked:
    X_new = build_feature_row()
    log_pred = pipe.predict(X_new)[0]
    price_pred = float(np.clip(np.expm1(log_pred), 0, None))

    st.success(f"Estimated nightly price: **${price_pred:,.0f}**")

    lo, hi = max(0, price_pred - rmse), price_pred + rmse
    st.caption(
        f"Typical model error is about ±${rmse} — realistic range roughly "
        f"**${lo:,.0f} – ${hi:,.0f}**."
    )

    with st.expander("See the exact input sent to the model"):
        st.dataframe(X_new.T.rename(columns={0: "value"}), use_container_width=True)
else:
    st.info("Fill in the listing details above and click **Predict nightly price**.")

st.divider()
st.caption(
    "DS605 Fundamentals of Machine Learning — Lab Assignment 4 | "
    "Model trained on Kaggle AB_NYC_2019 dataset."
)
