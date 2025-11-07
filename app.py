import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime

# ================
# APP CONFIGURATION
# ================
st.set_page_config(page_title="🔮 Insurance Renewal Predictor", layout="wide")
st.markdown(
    """
    <style>
    .main {background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%);}
    .stMetric {background: #f9fafb; border-radius: 15px; padding: 10px; box-shadow: 0 0 8px rgba(0,0,0,0.05);}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    h1 {color: #1e3a8a;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 Insurance Renewal Prediction Dashboard")
st.markdown(
    """
    ### Predict which customers are most likely to renew their insurance policies  
    Upload a dataset or manually enter customer details to get instant predictions, insights, and visuals.
    """
)

# ==========================
# LOAD MODEL, SCALER, ENCODER
# ==========================
MODEL_PATH = "renewal_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    st.error("❌ Required `.pkl` files missing. Please ensure model, scaler, and encoder files are available.")
    st.stop()

try:
    expected_features = list(scaler.feature_names_in_)
except AttributeError:
    expected_features = [
        "perc_premium_paid_by_cash_credit", "age_in_days", "Income",
        "Count_3-6_months_late", "Count_6-12_months_late",
        "Count_more_than_12_months_late", "application_underwriting_score",
        "no_of_premiums_paid", "premium",
    ]

cat_cols = ["sourcing_channel", "residence_area_type"]

# ================
# INPUT PROCESSING
# ================
def prepare_input(df):
    df = df.copy()
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        else:
            df[col] = -1

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df[expected_features] = df[expected_features].fillna(df[expected_features].median())

    X_scaled = pd.DataFrame(scaler.transform(df[expected_features]), columns=expected_features)
    for col in cat_cols:
        X_scaled[col] = df[col].values

    return X_scaled


# ==================
# SIDEBAR SELECTION
# ==================
st.sidebar.header("⚙️ Prediction Mode")
mode = st.sidebar.radio("Choose an option:", ["🧍 Single Entry", "📂 Batch Upload (CSV)"])

# ==========================
# SINGLE ENTRY PREDICTION
# ==========================
if "Single" in mode:
    st.subheader("🧍 Enter Customer Details")

    with st.container():
        col1, col2, col3 = st.columns(3)
        perc_premium_paid_by_cash_credit = col1.number_input("Percentage of Premium Paid by Cash/Credit", 0.0, 1.0, 0.5)
        dob = col2.date_input("Date of Birth", date(1990, 1, 1))
        Income = col3.number_input("Annual Income (₹)", 0, 10000000, 500000)

        age_years = (date.today() - dob).days / 365.25
        age_in_days = age_years * 365.25
        st.info(f"🧓 Age calculated: **{age_years:.1f} years**")

    col1, col2, col3 = st.columns(3)
    Count_3_6_months_late = col1.number_input("Count (3–6 months late)", 0.0, 10.0, 0.0)
    Count_6_12_months_late = col2.number_input("Count (6–12 months late)", 0.0, 10.0, 0.0)
    Count_more_than_12_months_late = col3.number_input("Count (>12 months late)", 0.0, 10.0, 0.0)

    col1, col2, col3 = st.columns(3)
    application_underwriting_score = col1.number_input("Underwriting Score", 0.0, 100.0, 98.0)
    no_of_premiums_paid = col2.number_input("Premiums Paid", 0, 100, 10)
    premium = col3.number_input("Premium Amount (₹)", 0, 100000, 5000)

    sourcing_channel = st.selectbox(
        "Sourcing Channel",
        ["Agent / Advisor (A)", "Branch Office (B)", "Corporate / Bancassurance (C)", "Digital / Online (D)", "Telemarketing / Call Center (E)"]
    )
    residence_area_type = st.selectbox(
        "Residence Area Type", ["Urban (City / Town)", "Rural (Village / Non-urban)"]
    )

    channel_map = {"Agent / Advisor (A)": "A", "Branch Office (B)": "B", "Corporate / Bancassurance (C)": "C", "Digital / Online (D)": "D", "Telemarketing / Call Center (E)": "E"}
    area_map = {"Urban (City / Town)": "Urban", "Rural (Village / Non-urban)": "Rural"}

    input_data = pd.DataFrame({
        "perc_premium_paid_by_cash_credit": [perc_premium_paid_by_cash_credit],
        "age_in_days": [age_in_days],
        "Income": [Income],
        "Count_3-6_months_late": [Count_3_6_months_late],
        "Count_6-12_months_late": [Count_6_12_months_late],
        "Count_more_than_12_months_late": [Count_more_than_12_months_late],
        "application_underwriting_score": [application_underwriting_score],
        "no_of_premiums_paid": [no_of_premiums_paid],
        "sourcing_channel": [channel_map[sourcing_channel]],
        "residence_area_type": [area_map[residence_area_type]],
        "premium": [premium],
    })

    X_final = prepare_input(input_data)

    if st.button("🔍 Predict Renewal Probability"):
        try:
            prob = model.predict_proba(X_final)[:, 1][0]
            st.success(f"**Predicted Renewal Probability: {prob:.2%}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Gauge Chart
        gauge = go.Figure(go.Indicator(
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
        st.plotly_chart(gauge, use_container_width=True)

        # Pie visualization
        pie = px.pie(
            names=["Will Renew", "Won’t Renew"],
            values=[prob, 1 - prob],
            color_discrete_sequence=["#22c55e", "#ef4444"],
            title="Renewal Likelihood Breakdown"
        )
        st.plotly_chart(pie, use_container_width=True)


# ==========================
# BATCH UPLOAD PREDICTION
# ==========================
else:
    st.subheader("📂 Upload Customer Data (CSV)")
    st.info("Include either `Date_of_Birth` or `age_in_days` column for accurate results.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Date_of_Birth" in df.columns:
            today = date.today()
            df["age_in_days"] = df["Date_of_Birth"].apply(
                lambda d: (today - pd.to_datetime(d).date()).days if pd.notnull(d) else np.nan
            )

        df = df.dropna(subset=["age_in_days"])
        X_final = prepare_input(df)

        try:
            df["Renewal_Probability"] = model.predict_proba(X_final)[:, 1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.success("✅ Predictions generated successfully!")
        st.dataframe(df.head(10))

        col1, col2 = st.columns(2)
        avg_prob = df["Renewal_Probability"].mean()
        high_prob_pct = (df["Renewal_Probability"] > 0.6).mean() * 100

        col1.metric("📊 Average Renewal Probability", f"{avg_prob:.2%}")
        col2.metric("💡 Likely Renewers (>60%)", f"{high_prob_pct:.1f}%")

        # Histogram of probabilities
        fig = px.histogram(df, x="Renewal_Probability", nbins=20,
                           title="Distribution of Renewal Probabilities",
                           color_discrete_sequence=["#2563eb"])
        st.plotly_chart(fig, use_container_width=True)

        # Donut visualization
        donut = px.pie(df, names=(df["Renewal_Probability"] > 0.6).map({True: "Likely Renewers", False: "Unlikely"}),
                       hole=0.4, title="Renewal Readiness Summary")
        st.plotly_chart(donut, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv, "renewal_predictions.csv", "text/csv")

    else:
        st.info("📤 Please upload a valid CSV file to begin predictions.")

