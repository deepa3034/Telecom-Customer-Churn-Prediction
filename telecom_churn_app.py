# CODE 16: Complete Telecom Customer Churn Prediction App

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. APP SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telecom Customer Churn Prediction")

st.write(
    "Enter customer information to predict churn risk "
    "using a trained Logistic Regression model."
)

st.divider()


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

folder = Path(__file__).resolve().parent

model = joblib.load(folder / "telecom_churn_model.pkl")
threshold = joblib.load(folder / "churn_threshold.pkl")
feature_columns = joblib.load(folder / "model_features.pkl")


# --------------------------------------------------
# 3. CUSTOMER INFORMATION FORM
# --------------------------------------------------

st.subheader("Customer Information")

with st.form("customer_form"):

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    predict_button = st.form_submit_button(
        "Predict Customer Churn"
    )


# --------------------------------------------------
# 4. PREPARE CUSTOMER DATA
# --------------------------------------------------

if predict_button:

    customer_data = {
        "tenure": tenure,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "InternetService": internet_service,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "Partner": partner,
        "Dependents": dependents
    }

    customer_df = pd.DataFrame([customer_data])

    # Check that all required model inputs are present.
    missing_features = [
        col for col in feature_columns
        if col not in customer_df.columns
    ]

    if missing_features:

        st.error(
            "The saved model requires additional inputs: "
            + ", ".join(missing_features)
        )

        st.stop()

    # Match the exact training feature order
    customer_df = customer_df[feature_columns]


    # --------------------------------------------------
    # 5. MAKE PREDICTION
    # --------------------------------------------------

    churn_probability = model.predict_proba(
        customer_df
    )[0][1]

    predicted_churn = churn_probability >= threshold

    st.divider()

    st.subheader("Prediction Results")

    st.metric(
        "Predicted Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )

    if predicted_churn:

        st.error("🔴 Customer At Risk of Churning")

        st.write(
            "This customer may benefit from a "
            "retention review."
        )

    else:

        st.success("🟢 Customer Not Flagged")

        st.write(
            "This customer's predicted churn probability "
            "is below the selected risk threshold."
        )

    st.info(
        f"Risk Classification Threshold: "
        f"{threshold * 100:.0f}%"
    )

    st.caption(
        "This is a Machine Learning prediction, "
        "not a guaranteed future outcome."
    )
