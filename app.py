
import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load Model, Scaler, Features
# ------------------------------
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
top_features = joblib.load("features.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

# ------------------------------
# User Inputs
# ------------------------------
input_data = {}

for feature in top_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([[input_data[f] for f in top_features]], columns=top_features)
    
    input_scaled = scaler.transform(input_df)

    # Use probability instead of direct predict
    proba = model.predict_proba(input_scaled)[0][1]

    st.write("Heart Disease Probability:", round(proba, 3))

    if proba > 0.5:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
