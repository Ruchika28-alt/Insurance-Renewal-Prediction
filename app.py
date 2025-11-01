import streamlit as st
import pandas as pd
import joblib
import os

st.title("🔮 Insurance Renewal Prediction App")
st.write("Upload customer data to predict renewal probabilities.")

# --- Load models safely ---
MODEL_PATH = os.path.join("models", "renewal_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
ENCODER_PATH = os.path.join("models", "encoder.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    st.error("Model, scaler, or encoder file not found. Please check 'models/' folder.")
    st.stop()

# --- File uploader ---
uploaded_file = st.file_uploader("Upload test.csv", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data Preview", data.head())

    # --- Drop ID if present ---
    X = data.drop(columns=['id'], errors='ignore')

    # --- Handle categorical encoding ---
    cat_cols = ['sourcing_channel', 'residence_area_type']
    for col in cat_cols:
        if col in X.columns:
            X[col] = encoder.transform(X[col])

    # --- Handle missing values ---
    X = X.fillna(X.median())

    # --- Scale numerical features ---
    X_scaled = scaler.transform(X)

    # --- Make predictions ---
    preds = model.predict_proba(X_scaled)[:, 1]

    # --- Add results ---
    data['Renewal_Probability'] = preds

    st.write("### 🔍 Renewal Probability per Customer")
    st.dataframe(data[['id', 'Renewal_Probability']])

    # --- Visualization ---
    st.bar_chart(data['Renewal_Probability'])

    # --- Download ---
    st.download_button(
        label="Download Predictions",
        data=data.to_csv(index=False),
        file_name="renewal_predictions.csv",
        mime="text/csv"
    )
