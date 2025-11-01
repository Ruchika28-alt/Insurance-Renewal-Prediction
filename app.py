import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🔮 Insurance Renewal Prediction", layout="wide")

st.title("🔮 Insurance Renewal Prediction App")
st.write("Upload customer data or enter details manually to predict renewal probabilities.")

# ======================
# Load trained components
# ======================
MODEL_PATH = "renewal_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)  # dictionary of label encoders
except FileNotFoundError:
    st.error("❌ Model, scaler, or encoder file not found. Make sure all .pkl files are in the same folder as app.py.")
    st.stop()

# ======================
# Sidebar - Mode Selection
# ======================
st.sidebar.header("⚙️ Select Prediction Mode")
mode = st.sidebar.radio("Choose an option:", ["Single Entry", "Batch Upload (CSV)"])

# ======================
# Single Entry Mode
# ======================
if mode == "Single Entry":
    st.subheader("🧍‍♀️ Enter Customer Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income", min_value=0, max_value=10000000, value=500000)
    premium = st.number_input("Premium Amount", min_value=0, max_value=500000, value=20000)
    claims = st.number_input("Number of Claims", min_value=0, max_value=20, value=0)
    interaction = st.number_input("Customer Interaction Score", min_value=0.0, max_value=1.0, value=0.5)
    channel = st.selectbox("Sourcing Channel", ["A", "B", "C", "D", "E"])
    residence = st.selectbox("Residence Area Type", ["Urban", "Rural"])

    input_dict = {
        "age": [age],
        "income": [income],
        "premium": [premium],
        "claims": [claims],
        "interaction": [interaction],
        "sourcing_channel": [channel],
        "residence_area_type": [residence],
    }

    X = pd.DataFrame(input_dict)

    # Encode categorical variables safely
    cat_cols = ["sourcing_channel", "residence_area_type"]
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            X[col] = X[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Scale numerical features
    try:
        if hasattr(model, "coef_"):  # Logistic Regression
            X_scaled = scaler.transform(X)
            pred_prob = model.predict_proba(X_scaled)[:, 1][0]
        else:  # Random Forest
            pred_prob = model.predict_proba(X)[:, 1][0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    st.markdown("---")
    st.success(f"**Predicted Renewal Probability:** {pred_prob:.2%}")

    if pred_prob > 0.6:
        st.info("✅ This customer is likely to renew their policy.")
    else:
        st.warning("⚠️ This customer might not renew. Consider follow-up actions.")

# ======================
# Batch Upload Mode
# ======================
else:
    st.subheader("📂 Upload Customer Data (CSV)")

    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        # Drop ID if present
        X = data.drop(columns=["id"], errors="ignore")
        X = X.fillna(X.median(numeric_only=True))

        # Encode categorical columns
        cat_cols = ["sourcing_channel", "residence_area_type"]
        for col in cat_cols:
            if col in X.columns:
                le = encoders.get(col)
                if le:
                    X[col] = X[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        # Predictions
        try:
            if hasattr(model, "coef_"):  # Logistic Regression
                X_scaled = scaler.transform(X)
                preds = model.predict_proba(X_scaled)[:, 1]
            else:
                preds = model.predict_proba(X)[:, 1]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        data["Renewal_Probability"] = preds

        st.subheader("🔍 Predicted Renewal Probabilities")
        st.dataframe(data.head(20))

        st.subheader("📈 Renewal Probability Distribution")
        st.bar_chart(data["Renewal_Probability"])

        # Download button
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name="renewal_predictions.csv",
            mime="text/csv",
        )

    else:
        st.info("👆 Upload a CSV file to predict in bulk.")
