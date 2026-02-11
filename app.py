import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------
# Load Model Files
# ------------------------------
MODEL_PATH = "heart_model (1).pkl"
SCALER_PATH = "scaler (1).pkl"
FEATURES_PATH = "features (1).pkl"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH)):
    st.error("❌ Model files not found! Put heart_model.pkl, scaler.pkl, features.pkl in the same folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
top_features = joblib.load(FEATURES_PATH)

# ------------------------------
# App UI
# ------------------------------
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

    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader(f"Heart Disease Probability: {round(proba * 100, 2)}%")
    if proba > 0.5:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")


