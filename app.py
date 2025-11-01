import streamlit as st
import pandas as pd
import joblib
import os

# Load model and scaler safely
MODEL_PATH = os.path.join("models", "renewal_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

st.title("🔮 Insurance Renewal Prediction")
st.write("Upload customer data to predict renewal probabilities")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'renewal_model.pkl' and 'scaler.pkl' are in the 'models/' folder.")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", data.head())

    X = data.drop(columns=['id'], errors='ignore')

    # Check if logistic regression
    if hasattr(model, 'coef_'):
        X_scaled = scaler.transform(X)
        preds = model.predict_proba(X_scaled)[:,1]
    else:
        preds = model.predict_proba(X)[:,1]

    data['Renewal_Probability'] = preds
    st.write("### 🔍 Renewal Probability per Customer")
    st.dataframe(data[['id', 'Renewal_Probability']])

    st.download_button(
        label="Download Predictions",
        data=data.to_csv(index=False),
        file_name="renewal_predictions.csv",
        mime="text/csv"
    )

    st.bar_chart(data['Renewal_Probability'])
