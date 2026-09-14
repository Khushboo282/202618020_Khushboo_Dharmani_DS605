import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Add src to system path so custom transformer can unpickle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import AirbnbFeatureEngineer

# Page Configuration
st.set_page_config(
    page_title="NYC Airbnb Price Prediction | DS605 Lab 4",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, simple CSS styling
st.markdown("""
    <style>
    .prediction-box {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 10px;
    }
    .predicted-price {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e40af;
        margin: 5px 0;
    }
    .price-range {
        font-size: 1rem;
        color: #475569;
    }
    .tier-tag {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
        background-color: #e0e7ff;
        color: #3730a3;
    }
    </style>
""", unsafe_allow_html=True)

# Load pipeline and metadata
@st.cache_resource
def load_pipeline():
    pipeline_path = os.path.join(CURRENT_DIR, 'models', 'pipeline.joblib')
    if not os.path.exists(pipeline_path):
        return None
    return joblib.load(pipeline_path)

@st.cache_data
def load_metadata():
    meta_path = os.path.join(CURRENT_DIR, 'models', 'neighbourhood_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_benchmark():
    csv_path = os.path.join(CURRENT_DIR, 'models', 'model_benchmark.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

pipeline = load_pipeline()
metadata = load_metadata()
benchmark_df = load_benchmark()

# Sidebar: Preset scenarios and author info
with st.sidebar:
    st.header("Airbnb Price Estimator")
    st.write("DS605: Fundamentals of Machine Learning — Lab 4")
    st.write("**Name:** Akanksha Dasani")
    st.write("**Roll No:** 202618062")
    st.write("---")
    
    st.subheader("Sample Listing Presets")
    st.caption("Select a sample scenario to populate the input form:")
    preset = st.radio(
        "Choose a listing:",
        [
            "Custom Input",
            "Midtown Manhattan Entire Apt",
            "Williamsburg Brooklyn Private Room",
            "Astoria Queens Budget Stay",
            "Harlem Historic Townhouse"
        ]
    )

st.title("NYC Airbnb Price Prediction")
st.write(
    "This web application predicts estimated nightly prices for Airbnb listings across "
    "New York City. The model is trained on the NYC 2019 Open Dataset using a tuned "
    "LightGBM regressor with domain-specific feature engineering."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Price Prediction", 
    "Data Exploration", 
    "Model Benchmark", 
    "Project Notes"
])

preset_values = {
    "Custom Input": {
        "borough": "Manhattan", "neighbourhood": "Midtown", "room_type": "Entire home/apt",
        "name": "Cozy Midtown Apartment near Times Square",
        "min_nights": 2, "reviews": 25, "rpm": 1.5, "host_count": 1, "avail": 180
    },
    "Midtown Manhattan Entire Apt": {
        "borough": "Manhattan", "neighbourhood": "Midtown", "room_type": "Entire home/apt",
        "name": "Spacious Midtown Loft near Central Park",
        "min_nights": 3, "reviews": 45, "rpm": 2.1, "host_count": 2, "avail": 240
    },
    "Williamsburg Brooklyn Private Room": {
        "borough": "Brooklyn", "neighbourhood": "Williamsburg", "room_type": "Private room",
        "name": "Cozy Sunny Room in Williamsburg",
        "min_nights": 2, "reviews": 85, "rpm": 3.4, "host_count": 1, "avail": 90
    },
    "Astoria Queens Budget Stay": {
        "borough": "Queens", "neighbourhood": "Astoria", "room_type": "Private room",
        "name": "Clean Room 15 mins to Manhattan",
        "min_nights": 1, "reviews": 12, "rpm": 0.8, "host_count": 1, "avail": 300
    },
    "Harlem Historic Townhouse": {
        "borough": "Manhattan", "neighbourhood": "Harlem", "room_type": "Entire home/apt",
        "name": "Brownstone Apartment in Central Harlem",
        "min_nights": 4, "reviews": 60, "rpm": 1.9, "host_count": 1, "avail": 120
    }
}

active = preset_values[preset]

# Tab 1: Price Prediction Form
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.subheader("Listing Details")
        borough_list = list(metadata.keys()) if metadata else ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        b_idx = borough_list.index(active["borough"]) if active["borough"] in borough_list else 0
        
        c1, c2 = st.columns(2)
        with c1:
            selected_borough = st.selectbox("Borough:", borough_list, index=b_idx)
        
        neigh_options = list(metadata.get(selected_borough, {}).keys()) if selected_borough in metadata else ['Midtown']
        n_idx = neigh_options.index(active["neighbourhood"]) if active["neighbourhood"] in neigh_options else 0
        
        with c2:
            selected_neighbourhood = st.selectbox("Neighbourhood:", neigh_options, index=n_idx)

        c3, c4 = st.columns(2)
        with c3:
            room_options = ['Entire home/apt', 'Private room', 'Shared room']
            selected_room = st.selectbox("Room Type:", room_options, index=room_options.index(active["room_type"]))
        with c4:
            listing_name = st.text_input("Listing Title:", value=active["name"])

        st.subheader("Booking & Availability")
        c5, c6, c7 = st.columns(3)
        with c5:
            min_nights = st.number_input("Minimum Nights:", min_value=1, max_value=365, value=active["min_nights"])
        with c6:
            host_count = st.number_input("Host Listings Count:", min_value=1, max_value=350, value=active["host_count"])
        with c7:
            avail_365 = st.slider("Availability (Days / Year):", min_value=0, max_value=365, value=active["avail"])

        st.subheader("Reviews")
        c8, c9 = st.columns(2)
        with c8:
            num_reviews = st.number_input("Total Number of Reviews:", min_value=0, max_value=700, value=active["reviews"])
        with c9:
            rpm = st.number_input("Reviews Per Month:", min_value=0.0, max_value=60.0, value=float(active["rpm"]), step=0.1)

        # Coordinate Auto-Fill
        neigh_info = metadata.get(selected_borough, {}).get(selected_neighbourhood, {'lat': 40.7580, 'lon': -73.9855})
        with st.expander("Geographic Coordinates (Auto-filled from neighbourhood)"):
            c_lat, c_lon = st.columns(2)
            with c_lat:
                lat = st.number_input("Latitude:", value=float(neigh_info['lat']), format="%.5f")
            with c_lon:
                lon = st.number_input("Longitude:", value=float(neigh_info['lon']), format="%.5f")

    with col_result:
        st.subheader("Estimated Nightly Price")
        
        if pipeline is None:
            st.error("Model pipeline could not be loaded. Please ensure models/pipeline.joblib exists.")
        else:
            input_df = pd.DataFrame([{
                'neighbourhood_group': selected_borough,
                'neighbourhood': selected_neighbourhood,
                'room_type': selected_room,
                'name': listing_name,
                'latitude': lat,
                'longitude': lon,
                'minimum_nights': min_nights,
                'number_of_reviews': num_reviews,
                'reviews_per_month': rpm,
                'calculated_host_listings_count': host_count,
                'availability_365': avail_365
            }])

            # Predict log price and convert to dollars
            pred_log = pipeline.predict(input_df)[0]
            pred_price = max(10.0, float(np.expm1(pred_log)))

            # 80% confidence bound using test MAE (~$45.09)
            lower_bound = max(10.0, pred_price - 36.0)
            upper_bound = pred_price + 36.0

            if pred_price < 80:
                tier_label = "Budget Listing"
            elif pred_price < 180:
                tier_label = "Moderate Listing"
            elif pred_price < 350:
                tier_label = "Premium Listing"
            else:
                tier_label = "Luxury Listing"

            st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 1rem; color: #64748b;">Estimated Price</div>
                    <div class="predicted-price">${pred_price:.2f}</div>
                    <div class="price-range">Expected Range: <b>${lower_bound:.0f} &ndash; ${upper_bound:.0f}</b> / night</div>
                    <div class="tier-tag">{tier_label}</div>
                </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.write("**Listing Location Map:**")
            map_data = pd.DataFrame([{'lat': lat, 'lon': lon}])
            st.map(map_data, zoom=12)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Neighbourhood Median", f"${neigh_info.get('median_price', 120):.0f}")
            with col_m2:
                dist = np.sqrt((lat - 40.7580)**2 + (lon - (-73.9855))**2) * 111.0
                st.metric("Distance to Midtown", f"{dist:.1f} km")

# Tab 2: Exploratory Visualizations
with tab2:
    st.subheader("Exploratory Data Analysis")
    st.write("Visualizations generated from our analysis of the 48,895 NYC Airbnb listings:")

    c_eda1, c_eda2 = st.columns(2)
    with c_eda1:
        st.write("**1. Raw vs Log-Transformed Price Distribution**")
        st.caption("Applying log(1 + price) removes the severe right skew and normalizes the target.")
        img_p1 = os.path.join(CURRENT_DIR, 'reports', 'figures', 'price_distribution.png')
        if os.path.exists(img_p1):
            st.image(img_p1, use_container_width=True)
    with c_eda2:
        st.write("**2. Nightly Price by Room Type**")
        st.caption("Entire homes command more than double the price of private or shared rooms.")
        img_p2 = os.path.join(CURRENT_DIR, 'reports', 'figures', 'room_type_prices.png')
        if os.path.exists(img_p2):
            st.image(img_p2, use_container_width=True)

    st.write("---")
    c_eda3, c_eda4 = st.columns(2)
    with c_eda3:
        st.write("**3. Average Price by NYC Borough**")
        st.caption("Manhattan is the most expensive borough, followed by Brooklyn.")
        img_p3 = os.path.join(CURRENT_DIR, 'reports', 'figures', 'borough_prices.png')
        if os.path.exists(img_p3):
            st.image(img_p3, use_container_width=True)
    with c_eda4:
        st.write("**4. Correlation Heatmap**")
        st.caption("Distance to Midtown shows a clear negative correlation with price.")
        img_p4 = os.path.join(CURRENT_DIR, 'reports', 'figures', 'correlation_matrix.png')
        if os.path.exists(img_p4):
            st.image(img_p4, use_container_width=True)

    st.write("---")
    st.write("**5. Listings Plotted Over New York City Map**")
    img_p5 = os.path.join(CURRENT_DIR, 'reports', 'figures', 'nyc_map_price.png')
    if os.path.exists(img_p5):
        st.image(img_p5, use_container_width=True)

# Tab 3: Model Benchmark
with tab3:
    st.subheader("Model Benchmark & Evaluation")
    st.write("We trained and compared 7 regression algorithms on an 80/20 train/test split:")
    
    if benchmark_df is not None:
        st.dataframe(benchmark_df, use_container_width=True)

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.write("**Train vs Test R2 (Overfitting Check)**")
        img_b1 = os.path.join(CURRENT_DIR, 'reports', 'figures', 'model_comparison.png')
        if os.path.exists(img_b1):
            st.image(img_b1, use_container_width=True)
    with c_b2:
        st.write("**Top Feature Importances (Tuned LightGBM)**")
        img_b2 = os.path.join(CURRENT_DIR, 'reports', 'figures', 'feature_importance.png')
        if os.path.exists(img_b2):
            st.image(img_b2, use_container_width=True)

    st.write("---")
    st.write("**Actual vs Predicted Prices & Residual Distribution:**")
    img_b3 = os.path.join(CURRENT_DIR, 'reports', 'figures', 'actual_vs_predicted_and_residuals.png')
    if os.path.exists(img_b3):
        st.image(img_b3, use_container_width=True)

# Tab 4: Project Notes
with tab4:
    st.subheader("Project Summary & Notes")
    st.write("""
    ### Assignment Details
    - **Course:** DS605: Fundamentals of Machine Learning
    - **Lab Assignment:** 4 (Airbnb Price Prediction)
    - **Student:** Akanksha Dasani (Roll No: 202618062)

    ### Main Steps Followed
    1. **Data Cleaning:** Filled 10,052 missing values in `reviews_per_month` with 0.0 because they strictly corresponded to listings with 0 total reviews. Dropped uninformative ID columns.
    2. **Outlier Filtering & Log Transform:** Removed 11 records with price <= $0 and filtered extreme luxury outliers above $1,000 to keep the loss stable. Applied `log(1 + price)` to normalize target skewness.
    3. **Feature Engineering:** Calculated distances to Midtown Manhattan (Times Square) and Wall Street, capped minimum nights at 30, flagged multi-listing commercial hosts, and grouped rare neighbourhoods into 'Other'.
    4. **Model Comparison:** Evaluated Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, LightGBM, and XGBoost. Found that Random Forest overfits (train R2 = 0.798 vs test R2 = 0.647), while LightGBM offered the best balance.
    5. **Tuning:** Tuned LightGBM using 3-fold GridSearchCV, achieving a test R2 of 0.6540 and MAE of $45.09.

    ### Real-World Limitations
    - **Missing Amenities:** Specific listing features like air conditioning, elevators, pools, or scenic views are not in the dataset, but they strongly affect real prices.
    - **NYC Regulation Changes:** The dataset is from 2019. In late 2023, New York City introduced Local Law 18, which strictly banned unhosted short-term rentals under 30 days.
    - **Seasonality:** The data is a static snapshot without booking dates, so peak holiday surges (Christmas, New Year's Eve) cannot be captured.
    """)
