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

# Sidebar for mode
st.sidebar.header("⚙️ Select Prediction Mode")
mode = st.sidebar.radio("Choose an option:", ["Single Entry", "Batch Upload (CSV)"])

# ======================
# SINGLE ENTRY MODE
# ======================
if mode == "Single Entry":
    st.subheader("🧍 Enter Customer Details")

    perc_premium_paid_by_cash_credit = st.number_input("Percentage of Premium Paid by Cash/Credit", 0.0, 1.0, 0.5)
    age_in_days = st.number_input("Age in Days", min_value=5000, max_value=40000, value=15000)
    Income = st.number_input("Income", min_value=0, max_value=10000000, value=300000)
    Count_3_6_months_late = st.number_input("Count (3-6 months late)", min_value=0.0, max_value=10.0, value=0.0)
    Count_6_12_months_late = st.number_input("Count (6-12 months late)", min_value=0.0, max_value=10.0, value=0.0)
    Count_more_than_12_months_late = st.number_input("Count (>12 months late)", min_value=0.0, max_value=10.0, value=0.0)
    application_underwriting_score = st.number_input("Application Underwriting Score", min_value=0.0, max_value=100.0, value=99.0)
    no_of_premiums_paid = st.number_input("Number of Premiums Paid", min_value=0, max_value=100, value=10)
    sourcing_channel = st.selectbox("Sourcing Channel", ["A", "B", "C", "D", "E"])
    residence_area_type = st.selectbox("Residence Area Type", ["Urban", "Rural"])
    premium = st.number_input("Premium Amount", min_value=0, max_value=100000, value=5000)

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

    # Encode categorical columns
    cat_cols = ["sourcing_channel", "residence_area_type"]
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            input_data[col] = input_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Scale numerical features
    num_cols = [c for c in input_data.columns if c not in cat_cols]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

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

        # Encode categorical columns
        cat_cols = ["sourcing_channel", "residence_area_type"]
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                X[col] = X[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        # Scale numeric
        num_cols = [c for c in X.columns if c not in cat_cols]
        X[num_cols] = scaler.transform(X[num_cols])

        # Predict
        preds = model.predict_proba(X)[:, 1]
        data["Renewal_Probability"] = preds

        st.subheader("🔍 Predicted Renewal Probabilities")
        st.dataframe(data.head(10))
        st.bar_chart(data["Renewal_Probability"])

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "renewal_predictions.csv", "text/csv")

    else:
        st.info("👆 Upload a CSV file to predict renewal probabilities in bulk.")
