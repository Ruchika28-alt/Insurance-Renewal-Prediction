import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Insurance Policy Renewal Prediction")
st.write("Upload customer data to predict renewal probabilities")

# Load trained model and scaler
model = joblib.load("renewal_model.pkl")
scaler = joblib.load("scaler.pkl")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", data.head())

    # Drop id if present
    X = data.drop(columns=['id'], errors='ignore')

    # Apply scaling if Logistic Regression
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
