import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model
model = pickle.load(open('catboost_model.pkl', 'rb'))

st.title("Heart Attack Prediction App ❤️")
st.write("Enter patient details to predict risk")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
thalach = st.number_input("Max Heart Rate", value=150)
oldpeak = st.number_input("Oldpeak", value=1.0)

sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])

# ADD THESE (IMPORTANT)
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):

    # BEST METHOD (no order issues)
    input_dict = {
        "cp": cp,
        "thalach": thalach,
        "ca": ca,
        "oldpeak": oldpeak,
        "chol": chol,
        "age": age,
        "exang": exang,
        "thal": thal
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)

    st.write("Prediction value:", prediction[0])  # Debug

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")
