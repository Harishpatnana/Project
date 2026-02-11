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
input_data['thal'] = st.number_input("Thal", min_value=3, max_value=7, value=4)
input_data['ca'] = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)
input_data['cp'] = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=1)
input_data['exang'] = st.number_input("Exercise Induced Angina (exang)", min_value=0, max_value=1, value=0)
input_data['thalach'] = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=71, max_value=202, value=150)
input_data['oldpeak'] = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, format="%.1f")
input_data['age'] = st.number_input("Age", min_value=29, max_value=77, value=54)
input_data['trestbps'] = st.number_input("Resting Blood Pressure (trestbps)", min_value=94, max_value=200, value=130)
input_data['chol'] = st.number_input("Cholesterol (chol)", min_value=126, max_value=564, value=246)
input_data['slope'] = st.number_input("Slope of Peak Exercise ST Segment (slope)", min_value=0, max_value=2, value=1)


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


