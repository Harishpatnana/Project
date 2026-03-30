import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('catboost_model.pkl', 'rb'))

st.title("Heart Attack Prediction App ❤️")
st.write("Enter patient details")

# ===== INPUTS =====

age = st.number_input("Age", 1, 120, 30)
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
thalach = st.number_input("Max Heart Rate", value=150)
oldpeak = st.number_input("Oldpeak", value=1.0)

sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])

ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

fbs = st.selectbox("Fasting Blood Sugar > 120 (0/1)", [0, 1])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
slope = st.selectbox("Slope (0–2)", [0, 1, 2])

# ===== PREDICTION =====

if st.button("Predict"):

    # Create full input dictionary
    input_dict = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "sex": sex,
        "cp": cp,
        "exang": exang,
        "ca": ca,
        "thal": thal,
        "fbs": fbs,
        "restecg": restecg,
        "slope": slope
    }

    input_df = pd.DataFrame([input_dict])

    # 🔥 IMPORTANT: Match EXACT training features
    top_features = ['cp', 'thalach', 'ca', 'oldpeak', 'chol', 'age', 'exang', 'thal']  # UPDATE if needed

    input_df = input_df[top_features]

    # Prediction
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")
