import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="🔮 Insurance Renewal Prediction", layout="wide")

st.title("🔮 Insurance Renewal Prediction App")
st.write("Upload customer data to predict renewal probabilities.")

# ======================
# Load trained components
# ======================
MODEL_PATH = "renewal_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)   # dict of label encoders
except FileNotFoundError:
    st.error("❌ Model, scaler, or encoder file not found. Make sure all .pkl files are in the same folder as app.py.")
    st.stop()

# ======================
# File upload section
# ======================
uploaded_file = st.file_uploader("📂 Upload test.csv file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())

    # Drop ID if present
    X = data.drop(columns=['id'], errors='ignore')

    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))

    # Encode categorical columns
    cat_cols = ['sourcing_channel', 'residence_area_type']
    for col in cat_cols:
        if col in X.columns:
            le = encoders.get(col)
            if le:
                # Map unknown categories safely
                X[col] = X[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            else:
                st.warning(f"No encoder found for {col}. Column skipped.")

    # ======================
    # Prediction
    # ======================
    try:
        if hasattr(model, 'coef_'):  # Logistic Regression
            X_scaled = scaler.transform(X)
            preds = model.predict_proba(X_scaled)[:, 1]
        else:  # Random Forest
            preds = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Combine with original data
    data['Renewal_Probability'] = preds

    st.subheader("🔍 Predicted Renewal Probabilities")
    st.dataframe(data[['id', 'Renewal_Probability']].head(20))

    # ======================
    # Visualization
    # ======================
    st.subheader("📈 Renewal Probability Distribution")
    st.bar_chart(data['Renewal_Probability'])

    # ======================
    # Download option
    # ======================
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv,
        file_name='renewal_predictions.csv',
        mime='text/csv'
    )

else:
    st.info("👆 Upload a CSV file to begin.")
