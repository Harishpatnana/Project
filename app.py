import streamlit as st
import numpy as np
import pickle

# Load your trained model
model = pickle.load(open('catboost_model.pkl', 'rb'))

st.title("Heart Attack Prediction App ❤️")

st.write("Enter patient details to predict risk")

# ===== INPUT FEATURES (EDIT THESE BASED ON YOUR top_features) =====

age = st.number_input("Age", min_value=1, max_value=120, value=30)
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
thalach = st.number_input("Max Heart Rate", value=150)
oldpeak = st.number_input("Oldpeak", value=1.0)

# Categorical inputs (adjust if needed)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])

# ===== PREDICTION =====

if st.button("Predict"):

    # Correct order (EXAMPLE — replace with YOUR order)
    input_data = np.array([[cp, thalach, ca, oldpeak, chol, age, exang, thal]])

    prediction = model.predict(input_data)

    st.write("Prediction value:", prediction[0])  # DEBUG

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")
