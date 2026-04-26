import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import shap
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

# --- LISTS ---
ALL_MANUFACTURERS = sorted([
    "acura", "alfa-romeo", "aston-martin", "audi", "bentley",
    "bmw", "buick", "cadillac", "chevrolet", "chrysler",
    "dacia", "dodge", "ferrari", "fiat", "ford", "genesis",
    "gmc", "harley-davidson", "honda", "hyundai", "infiniti",
    "jaguar", "jeep", "kia", "lamborghini", "land rover",
    "lexus", "lincoln", "lotus", "maserati", "mazda",
    "mclaren", "mercedes-benz", "mercury", "mini", "mitsubishi",
    "morgan", "nissan", "opel", "peugeot", "pontiac",
    "porsche", "ram", "renault", "rivian", "rolls-royce",
    "rover", "saturn", "scion", "seat", "skoda",
    "subaru", "tesla", "toyota", "volkswagen", "volvo",
    "other"
])

ALL_CONDITIONS    = ["excellent", "good", "like new", "new", "fair", "salvage", "unknown"]
ALL_CYLINDERS     = ["3 cylinders", "4 cylinders", "5 cylinders", "6 cylinders",
                     "8 cylinders", "10 cylinders", "12 cylinders", "other", "unknown"]
ALL_FUELS         = ["gas", "diesel", "electric", "hybrid", "other"]
ALL_TRANSMISSIONS = ["automatic", "manual", "other"]
ALL_DRIVES        = ["4wd", "fwd", "rwd", "unknown"]
ALL_TYPES         = ["SUV", "bus", "convertible", "coupe", "hatchback", "mini-van",
                     "offroad", "other", "pickup", "sedan", "truck", "van",
                     "wagon", "unknown"]
ALL_COLORS        = ["black", "blue", "brown", "custom", "green", "grey",
                     "orange", "purple", "red", "silver", "white", "yellow", "unknown"]
ALL_STATES        = sorted([
                     "ak","al","ar","az","ca","co","ct","dc","de","fl",
                     "ga","hi","ia","id","il","in","ks","ky","la","ma",
                     "md","me","mi","mn","mo","ms","mt","nc","nd","ne",
                     "nh","nj","nm","nv","ny","oh","ok","or","pa","ri",
                     "sc","sd","tn","tx","ut","va","vt","wa","wi","wv","wy"
                     ]) + ["other"]

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    base = "models"
    with open(f"{base}/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(f"{base}/xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(f"{base}/lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(f"{base}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{base}/cat_unique.json", "r") as f:
        cat_unique = json.load(f)
    with open(f"{base}/shap_values.pkl", "rb") as f:
        shap_values = pickle.load(f)
    with open(f"{base}/X_sample.pkl", "rb") as f:
        X_sample = pickle.load(f)
    return rf, xgb, lr, scaler, cat_unique, shap_values, X_sample

rf_model, xgb_model, lr_model, scaler, cat_unique, shap_values, X_sample = load_models()

# --- ENCODING HELPER ---
def safe_encode(col, val, cat_unique):
    known_vals = cat_unique[col]
    if val not in known_vals:
        val = "unknown" if "unknown" in known_vals else known_vals[0]
    le = LabelEncoder()
    le.fit(known_vals)
    return le.transform([val])[0], val

# --- SIDEBAR ---
st.sidebar.title("Car Price Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Models", "Price Prediction"]
)

# =============================================================
# PAGE 1 — HOME
# =============================================================
if page == "Home":
    st.title("Car Price Prediction — Craigslist Dataset")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Data", "426,880 rows")
    col2.metric("After Cleaning", "367,218 rows")
    col3.metric("Features Used", "14")
    col4.metric("Best R2", "0.8289")

    st.markdown("---")
    st.subheader("About the Dataset")
    st.markdown("""
    This application uses a dataset scraped from **Craigslist**, the largest
    second-hand vehicle marketplace in the United States. The dataset contains
    listings from all 50 states with details such as manufacturer, condition,
    odometer reading, fuel type, and more.

    The target variable is **price** — the listed sale price of each vehicle.
    """)

    st.subheader("Data Cleaning")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Columns removed (12):**
        Irrelevant or identifier columns such as
        url, VIN, image_url, description, county,
        region, lat, long, posting_date and size
        were dropped from the dataset.

        **Outlier filtering:**
        - Price: $500 — $150,000
        - Year: 1990 — 2022
        - Odometer: 0 — 300,000 miles
        """)
    with col2:
        st.markdown("""
        **Missing values:**
        - High missingness (>20%): filled with "unknown" category
        - Low missingness (<5%): filled with mode

        **Key finding:**
        After cleaning, price-year correlation
        jumped from -0.00 to +0.57, and
        price-odometer from 0.01 to -0.54.
        This confirms that outliers were masking
        the true relationships in the data.
        """)

    st.markdown("---")
    st.subheader("Feature Engineering")
    fe_data = pd.DataFrame({
        "Feature": ["vehicle_age", "age_odometer_ratio", "is_luxury", "is_clean_title"],
        "Formula": [
            "2024 - year",
            "odometer / (vehicle_age + 1)",
            "1 if brand is luxury else 0",
            "1 if title_status == clean else 0"
        ],
        "Description": [
            "Age of the vehicle in years",
            "Average annual mileage — measures wear rate",
            "Binary flag for luxury brands (BMW, Audi, Mercedes etc.)",
            "Binary flag for clean title status"
        ]
    })
    st.dataframe(fe_data, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Results Overview")
    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost v2"],
        "MAE ($)": [5646, 3439, 3803],
        "RMSE ($)": [8610, 6078, 6437],
        "R2": [0.6567, 0.8289, 0.8081]
    })
    st.dataframe(results, use_container_width=True)

# =============================================================
# PAGE 2 — MODELS
# =============================================================
elif page == "Models":
    st.title("Models")
    st.markdown("---")

    st.markdown("""
    Three machine learning models from different algorithm families were trained
    and compared on this dataset. Each model represents a different approach
    to the regression problem.
    """)

    st.subheader("1. Linear Regression — Baseline")
    st.markdown("""
    Linear Regression is the simplest model used in this project. It assumes
    a linear relationship between the input features and the target variable (price).
    Each feature is assigned a coefficient that represents its contribution to the
    predicted price.

    **Why include it?**
    Every machine learning project needs a baseline. Without it, there is no
    reference point to measure how much the more complex models actually improve.

    **Limitation:**
    The relationship between vehicle price and features like age or mileage is
    not strictly linear. This model also produced negative price predictions for
    some inputs, which is not meaningful in practice.
    """)
    st.markdown("---")

    st.subheader("2. Random Forest — Ensemble / Bagging")
    st.markdown("""
    Random Forest builds a large number of decision trees (100 in this project),
    each trained on a random subset of the data and features. The final prediction
    is the average of all individual tree predictions.

    **Why is it powerful?**
    Unlike Linear Regression, Random Forest can capture non-linear relationships
    and complex interactions between features. For example, it can learn that
    a 5-year-old BMW behaves very differently from a 5-year-old Honda in terms
    of price — something a linear model cannot express.

    **Parameters used:**
    n_estimators = 100, max_depth = 15, min_samples_leaf = 5
    """)
    st.markdown("---")

    st.subheader("3. XGBoost — Ensemble / Boosting")
    st.markdown("""
    XGBoost (Extreme Gradient Boosting) also builds multiple decision trees,
    but unlike Random Forest, each tree is trained to correct the errors made
    by the previous one. This sequential learning process makes it highly
    effective on structured tabular data.

    **Why include it?**
    XGBoost is considered state-of-the-art for tabular datasets and wins the
    majority of structured data competitions. It also includes built-in
    regularization (L1 and L2), which helps prevent overfitting.

    **Parameters used:**
    n_estimators = 500, learning_rate = 0.02, max_depth = 7,
    subsample = 0.8, colsample_bytree = 0.8
    """)
    st.markdown("---")

    st.subheader("Performance Comparison")
    st.markdown("""
    Three metrics are used to evaluate model performance:

    - **MAE (Mean Absolute Error):** Average dollar amount the model is off by.
    If MAE is $3,439 it means predictions are wrong by $3,439 on average.
    - **RMSE (Root Mean Squared Error):** Similar to MAE but penalizes large
    errors more heavily. Always higher than MAE.
    - **R2 (R-squared):** Proportion of price variance explained by the model.
    Ranges from 0 to 1. Higher is better — 0.83 means the model explains
    83% of the variation in price.
    """)

    models    = ["Linear Regression", "Random Forest", "XGBoost v2"]
    mae_vals  = [5646, 3439, 3803]
    rmse_vals = [8610, 6078, 6437]
    r2_vals   = [0.6567, 0.8289, 0.8081]
    colors    = ["#e74c3c", "#2ecc71", "#3498db"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(models, mae_vals, color=colors)
    axes[0].set_title("MAE — Lower is Better", fontweight="bold")
    axes[0].set_ylabel("Dollars ($)")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(mae_vals):
        axes[0].text(i, v+50, f"${v:,}", ha="center", fontweight="bold")

    axes[1].bar(models, rmse_vals, color=colors)
    axes[1].set_title("RMSE — Lower is Better", fontweight="bold")
    axes[1].set_ylabel("Dollars ($)")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v+50, f"${v:,}", ha="center", fontweight="bold")

    axes[2].bar(models, r2_vals, color=colors)
    axes[2].set_title("R2 Score — Higher is Better", fontweight="bold")
    axes[2].set_ylabel("R2")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=15)
    for i, v in enumerate(r2_vals):
        axes[2].text(i, v+0.01, f"{v:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(
        "Random Forest achieved the best performance across all metrics. "
        "MAE: $3,439 | RMSE: $6,078 | R2: 0.8289 — "
        "explaining 82.9% of price variance."
    )

    st.markdown("---")
    st.subheader("Why did Random Forest outperform XGBoost?")
    st.markdown("""
    XGBoost is generally considered superior for tabular data, yet Random Forest
    performed better here. The likely reasons are:

    - The dataset is large (293,774 training rows) — Random Forest scales well at this size
    - The dataset is heavily categorical (9 categorical columns) — Random Forest
    handles these naturally
    - XGBoost would likely close the gap with extensive hyperparameter tuning
    (e.g. via Optuna or GridSearchCV), but this falls outside the scope of this project

    This result itself is a meaningful finding worth discussing.
    """)

    st.markdown("---")
    st.subheader("Error Distribution")
    st.markdown("""
    | Model | Mean Error | Std Dev | 95% Interval |
    |---|---|---|---|
    | Linear Regression | -$29 | $8,610 | -$13,929 / +$18,106 |
    | Random Forest | -$29 | $6,078 | -$10,442 / +$11,916 |
    | XGBoost v2 | -$11 | $6,437 | -$10,624 / +$13,140 |

    All three models have a mean error close to zero, meaning none of them
    systematically over or under predict. The key difference is the spread:
    Random Forest has the narrowest error distribution, meaning its predictions
    are the most consistent.
    """)

# =============================================================
# PAGE 3 — PRICE PREDICTION
# =============================================================
elif page == "Price Prediction":
    st.title("Car Price Prediction")
    st.markdown("---")
    st.markdown("Enter vehicle details below to get a price estimate using the Random Forest model.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Manufacturer**")
        manufacturer_select = st.selectbox(
            "Select from list",
            ["Select..."] + ALL_MANUFACTURERS,
            key="mfr_select"
        )
        manufacturer_custom = st.text_input(
            "Not in list? Type here",
            placeholder="e.g. seat, dacia, skoda...",
            key="mfr_custom"
        )
        if manufacturer_custom.strip():
            manufacturer = manufacturer_custom.strip().lower()
        elif manufacturer_select != "Select...":
            manufacturer = manufacturer_select
        else:
            manufacturer = "other"

        condition    = st.selectbox("Condition", ALL_CONDITIONS)
        cylinders    = st.selectbox("Cylinders", ALL_CYLINDERS)
        fuel         = st.selectbox("Fuel Type", ALL_FUELS)
        transmission = st.selectbox("Transmission", ALL_TRANSMISSIONS)

    with col2:
        drive       = st.selectbox("Drive Type", ALL_DRIVES)
        type_       = st.selectbox("Vehicle Type", ALL_TYPES)
        paint_color = st.selectbox("Color", ALL_COLORS)
        state       = st.selectbox("State", ALL_STATES)

    with col3:
        vehicle_age = st.number_input(
            "Vehicle Age (years)",
            min_value=2,
            max_value=34,
            value=10,
            step=1
        )
        unit = st.radio(
            "Odometer Unit",
            ["Miles", "Kilometers"],
            horizontal=True
        )
        odometer_input = st.number_input(
            f"Odometer ({'miles' if unit == 'Miles' else 'km'})",
            min_value=0,
            max_value=500000 if unit == "Kilometers" else 300000,
            value=80000 if unit == "Kilometers" else 50000,
            step=1000
        )
        # Modele göndermeden önce km ise mile'a çevir
        odometer = odometer_input * 0.621371 if unit == "Kilometers" else odometer_input
        odometer = min(odometer, 300000)  # Model limitini aşmasın

    st.markdown("---")

    if st.button("Predict Price", use_container_width=True):

        luxury_brands = ["bmw", "mercedes-benz", "audi", "lexus",
                         "porsche", "jaguar", "land rover", "cadillac"]
        is_luxury          = 1 if manufacturer in luxury_brands else 0
        is_clean_title     = 1
        age_odometer_ratio = odometer / (vehicle_age + 1)

        state_val = "unknown" if state == "other" else state

        cat_cols_order = ["manufacturer", "condition", "cylinders", "fuel",
                          "transmission", "drive", "type", "paint_color", "state"]
        cat_vals_order = [manufacturer, condition, cylinders, fuel,
                          transmission, drive, type_, paint_color, state_val]

        encoded  = {}
        warnings = []

        for col, val in zip(cat_cols_order, cat_vals_order):
            enc_val, used_val = safe_encode(col, val, cat_unique)
            encoded[col] = enc_val
            if used_val != val:
                warnings.append(f"{col}: {val} mapped to {used_val}")

        if warnings:
            st.warning(
                "Some values were not seen during training and were mapped to the closest known category:\n\n" +
                "\n".join(warnings)
            )

        num_raw    = np.array([[odometer, vehicle_age, age_odometer_ratio]])
        num_scaled = scaler.transform(num_raw)

        input_df = pd.DataFrame([{
            "manufacturer":       encoded["manufacturer"],
            "condition":          encoded["condition"],
            "cylinders":          encoded["cylinders"],
            "fuel":               encoded["fuel"],
            "odometer":           num_scaled[0][0],
            "transmission":       encoded["transmission"],
            "drive":              encoded["drive"],
            "type":               encoded["type"],
            "paint_color":        encoded["paint_color"],
            "state":              encoded["state"],
            "vehicle_age":        num_scaled[0][1],
            "age_odometer_ratio": num_scaled[0][2],
            "is_luxury":          is_luxury,
            "is_clean_title":     is_clean_title
        }])

        prediction = rf_model.predict(input_df)[0]
        prediction = max(500, prediction)

        st.success(f"Estimated Price: ${prediction:,.0f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Price", f"${prediction:,.0f}")
        col2.metric("Model Used", "Random Forest")
        col3.metric("Model R2", "0.8289")

        st.markdown(f"""
        **Input Summary:**
        - Manufacturer: {manufacturer} | Condition: {condition} | Fuel: {fuel}
        - Vehicle Age: {vehicle_age} years | Odometer: {odometer:,} miles
        - Luxury Vehicle: {"Yes" if is_luxury else "No"}
        """)
