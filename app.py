import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🔮 Insurance Renewal Prediction", layout="wide")
st.title("🔮 Insurance Renewal Prediction App")
st.write("Upload customer data or enter details manually to predict renewal probabilities.")

# Load model, scaler, encoder
MODEL_PATH = "renewal_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    st.error("❌ Model, scaler, or encoder file not found. Ensure all .pkl files are in this folder.")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Select Prediction Mode")
mode = st.sidebar.radio("Choose an option:", ["Single Entry", "Batch Upload (CSV)"])

# Get feature names from scaler if available
try:
    scaler_features = scaler.feature_names_in_
except AttributeError:
    scaler_features = [
        "perc_premium_paid_by_cash_credit",
        "age_in_days",
        "Income",
        "Count_3-6_months_late",
        "Count_6-12_months_late",
        "Count_more_than_12_months_late",
        "application_underwriting_score",
        "no_of_premiums_paid",
        "premium"
    ]

cat_cols = ["sourcing_channel", "residence_area_type"]

# ======================
# SINGLE ENTRY MODE
# ======================
if mode == "Single Entry":
    st.subheader("🧍 Enter Customer Details")

    perc_premium_paid_by_cash_credit = st.number_input("Percentage of Premium Paid by Cash/Credit", 0.0, 1.0, 0.5)
    age_in_days = st.number_input("Age in Days", min_value=5000, max_value=40000, value=15000)
    Income = st.number_input("Income", min_value=0, max_value=10000000, value=300000)
    Count_3_6_months_late = st.number_input("Count (3-6 months late)", 0.0, 10.0, 0.0)
    Count_6_12_months_late = st.number_input("Count (6-12 months late)", 0.0, 10.0, 0.0)
    Count_more_than_12_months_late = st.number_input("Count (>12 months late)", 0.0, 10.0, 0.0)
    application_underwriting_score = st.number_input("Application Underwriting Score", 0.0, 100.0, 99.0)
    no_of_premiums_paid = st.number_input("Number of Premiums Paid", 0, 100, 10)
    sourcing_channel = st.selectbox("Sourcing Channel", ["A", "B", "C", "D", "E"])
    residence_area_type = st.selectbox("Residence Area Type", ["Urban", "Rural"])
    premium = st.number_input("Premium Amount", 0, 100000, 5000)

    input_data = pd.DataFrame({
        "perc_premium_paid_by_cash_credit": [perc_premium_paid_by_cash_credit],
        "age_in_days": [age_in_days],
        "Income": [Income],
        "Count_3-6_months_late": [Count_3_6_months_late],
        "Count_6-12_months_late": [Count_6_12_months_late],
        "Count_more_than_12_months_late": [Count_more_than_12_months_late],
        "application_underwriting_score": [application_underwriting_score],
        "no_of_premiums_paid": [no_of_premiums_paid],
        "sourcing_channel": [sourcing_channel],
        "residence_area_type": [residence_area_type],
        "premium": [premium],
    })

    # Encode categorical
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            input_data[col] = input_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Align columns to scaler
    for col in scaler_features:
        if col not in input_data.columns:
            input_data[col] = 0  # add missing feature if any

    input_data = input_data[[*scaler_features, *cat_cols]]

    # Scale numeric part
    input_data[scaler_features] = scaler.transform(input_data[scaler_features])

    # Predict
    prob = model.predict_proba(input_data)[:, 1][0]

    st.markdown("---")
    st.success(f"**Predicted Renewal Probability:** {prob:.2%}")
    if prob > 0.6:
        st.info("✅ This customer is likely to renew their policy.")
    else:
        st.warning("⚠️ This customer might not renew. Consider customer engagement strategies.")

# ======================
# BATCH UPLOAD MODE
# ======================
else:
    st.subheader("📂 Upload Customer Data (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        X = data.drop(columns=["id"], errors="ignore")
        X = X.fillna(X.median(numeric_only=True))

        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                X[col] = X[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        for col in scaler_features:
            if col not in X.columns:
                X[col] = 0

        X = X[[*scaler_features, *cat_cols]]
        X[scaler_features] = scaler.transform(X[scaler_features])

        preds = model.predict_proba(X)[:, 1]
        data["Renewal_Probability"] = preds

        st.subheader("🔍 Predicted Renewal Probabilities")
        st.dataframe(data.head(10))
        st.bar_chart(data["Renewal_Probability"])

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "renewal_predictions.csv", "text/csv")

    else:
        st.info("👆 Upload a CSV file to predict renewal probabilities in bulk.")
