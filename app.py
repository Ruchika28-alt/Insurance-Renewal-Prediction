import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime

# App config
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
    st.error("❌ Model, scaler, or encoder file not found. Ensure all .pkl files are present.")
    st.stop()

# Expected features
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
        "premium",
    ]

cat_cols = ["sourcing_channel", "residence_area_type"]

# ======================
# FUNCTION: CLEAN INPUT
# ======================
def prepare_input(df):
    df = df.copy()

    # Encode categoricals
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        else:
            df[col] = -1

    # Ensure all expected numeric columns exist
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Fill missing numeric with median
    df[expected_features] = df[expected_features].fillna(df[expected_features].median())

    # Scale numeric columns safely
    X_num_scaled = pd.DataFrame(
        scaler.transform(df[expected_features]),
        columns=expected_features
    )

    # Add encoded categoricals
    for col in cat_cols:
        X_num_scaled[col] = df[col].values

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

    perc_premium_paid_by_cash_credit = st.number_input(
        "Percentage of Premium Paid by Cash/Credit", 0.0, 1.0, 0.5
    )

    dob = st.date_input("Date of Birth", date(1990, 1, 1))
    today = date.today()
    age_years = (today - dob).days / 365.25
    age_in_days = age_years * 365.25
    st.info(f"🧓 Age calculated: **{age_years:.1f} years**")

    Income = st.number_input("Income", 0, 10000000, 300000)
    Count_3_6_months_late = st.number_input("Count (3-6 months late)", 0.0, 10.0, 0.0)
    Count_6_12_months_late = st.number_input("Count (6-12 months late)", 0.0, 10.0, 0.0)
    Count_more_than_12_months_late = st.number_input("Count (>12 months late)", 0.0, 10.0, 0.0)
    application_underwriting_score = st.number_input("Application Underwriting Score", 0.0, 100.0, 99.0)
    no_of_premiums_paid = st.number_input("Number of Premiums Paid", 0, 100, 10)

    # Full sourcing channel labels
    channel_map = {
        "Agent / Advisor (A)": "A",
        "Branch Office (B)": "B",
        "Corporate / Bancassurance (C)": "C",
        "Digital / Online (D)": "D",
        "Telemarketing / Call Center (E)": "E",
    }
    sourcing_channel = channel_map[
        st.selectbox("Sourcing Channel", list(channel_map.keys()))
    ]

    area_map = {
        "Urban (City / Town)": "Urban",
        "Rural (Village / Non-urban)": "Rural",
    }
    residence_area_type = area_map[
        st.selectbox("Residence Area Type", list(area_map.keys()))
    ]

    premium = st.number_input("Premium Amount", 0, 100000, 5000)

    # Prepare input
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

    # Predict safely
    try:
        prob = model.predict_proba(X_final)[:, 1][0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.success(f"**Predicted Renewal Probability:** {prob:.2%}")

    # Gauge visualization
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
# BATCH MODE
# ======================
else:
    st.subheader("📂 Upload Customer Data (CSV)")
    st.info("Your CSV should include either a `Date_of_Birth` column or `age_in_days`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Date_of_Birth" in df.columns:
            today = date.today()
            df["age_in_days"] = df["Date_of_Birth"].apply(
                lambda d: (today - pd.to_datetime(d).date()).days if pd.notnull(d) else np.nan
            )

        df = df.dropna(subset=["age_in_days"], how="any")
        st.dataframe(df.head())

        X_final = prepare_input(df)
        try:
            preds = model.predict_proba(X_final)[:, 1]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        df["Renewal_Probability"] = preds
        st.subheader("🔍 Predicted Renewal Probabilities")
        st.dataframe(df.head(10))

        # Visualization
        fig = px.histogram(df, x="Renewal_Probability", nbins=20, title="Distribution of Renewal Probabilities")
        st.plotly_chart(fig, use_container_width=True)

        avg_prob = df["Renewal_Probability"].mean()
        high_prob_pct = (df["Renewal_Probability"] > 0.6).mean() * 100

        col1, col2 = st.columns(2)
        col1.metric("Average Renewal Probability", f"{avg_prob:.2%}")
        col2.metric("Likely Renewers (>0.6)", f"{high_prob_pct:.1f}%")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "renewal_predictions.csv", "text/csv")
    else:
        st.info("👆 Upload a CSV file to start predictions.")
