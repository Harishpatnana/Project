import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("❤️ Heart Disease Prediction App")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")
