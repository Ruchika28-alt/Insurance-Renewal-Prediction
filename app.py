import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="🔮 Insurance Renewal Prediction", layout="wide")
st.title("🔮 Insurance Renewal Prediction App")
st.write("Upload customer data or enter details manually to predict renewal probabilities with visual insights.")

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

# Handle scaler feature names
try:
    expected_features = list(scaler.feature_names_in_)
except AttributeError:
    expected_features = [
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
# FUNCTION: CLEAN INPUT
# ======================
def prepare_input(df):
    """Align columns with scaler and encoder expectations."""
    df = df.copy()

    # Encode categorical columns
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Add missing numeric features
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected numeric features
    X_num = df[expected_features].copy()

    # Scale numeric safely
    X_num_scaled = pd.DataFrame(
        scaler.transform(X_num),
        columns=expected_features
    )

    # Combine numeric + categorical (if exist)
    for col in cat_cols:
        if col in df.columns:
            X_num_scaled[col] = df[col]

    return X_num_scaled


# ======================
# SIDEBAR MODE
# ======================
st.sidebar.header("⚙️ Select Prediction Mode")
mode = st.sidebar.radio("Choose an option:", ["Single Entry", "Batch Upload (CSV)"])

# ======================
# SINGLE ENTRY MODE
# ======================
if mode == "Single Entry":
    st.subheader("🧍 Enter Customer Details")

    perc_premium_paid_by_cash_credit = st.number_input("Percentage of Premium Paid by Cash/Credit", 0.0, 1.0, 0.5)
    age_in_days = st.number_input("Age in Days", 5000, 40000, 15000)
    Income = st.number_input("Income", 0, 10000000, 300000)
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

    X_final = prepare_input(input_data)
    prob = model.predict_proba(X_final)[:, 1][0]

    st.markdown("---")
    st.success(f"**Predicted Renewal Probability:** {prob:.2%}")

    # Gauge Visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Renewal Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if prob > 0.6 else "red"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 70], 'color': "#fff4cc"},
                {'range': [70, 100], 'color': "#d9ffcc"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ======================
# BATCH UPLOAD MODE
# ======================
else:
    st.subheader("📂 Upload Customer Data (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        X_final = prepare_input(data)
        preds = model.predict_proba(X_final)[:, 1]
        data["Renewal_Probability"] = preds

        st.subheader("🔍 Predicted Renewal Probabilities")
        st.dataframe(data.head(10))

        # Visualization
        fig1 = px.histogram(
            data,
            x="Renewal_Probability",
            nbins=20,
            title="Distribution of Renewal Probabilities",
            color_discrete_sequence=["#2b8cbe"]
        )
        st.plotly_chart(fig1, use_container_width=True)

        avg_prob = data["Renewal_Probability"].mean()
        high_prob_pct = (data["Renewal_Probability"] > 0.6).mean() * 100

        col1, col2 = st.columns(2)
        col1.metric("Average Renewal Probability", f"{avg_prob:.2%}")
        col2.metric("Customers Likely to Renew (>0.6)", f"{high_prob_pct:.1f}%")

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "renewal_predictions.csv", "text/csv")
    else:
        st.info("👆 Upload a CSV file to predict renewal probabilities in bulk.")
