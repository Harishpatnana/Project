import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('catboost_model.pkl', 'rb'))

st.title("Heart Attack Risk Prediction ❤️")

# ===== INPUTS (USE YOUR FEATURES) =====

cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
thalach = st.number_input("Max Heart Rate", value=150)
ca = st.selectbox("Number of Vessels (0–3)", [0,1,2,3])
oldpeak = st.number_input("Oldpeak", value=1.0)
chol = st.number_input("Cholesterol", value=200)
age = st.number_input("Age", 1, 120, 30)
exang = st.selectbox("Exercise Angina (0/1)", [0,1])
thal = st.selectbox("Thal (0–3)", [0,1,2,3])

# ===== PREDICT =====

if st.button("Predict"):

    input_df = pd.DataFrame([[cp, thalach, ca, oldpeak, chol, age, exang, thal]],
                            columns=['cp','thalach','ca','oldpeak','chol','age','exang','thal'])

    # 🔥 Get probability instead of class
    prob = model.predict_proba(input_df)[0][1]   # probability of class 1

    percentage = prob * 100

    st.subheader(f"Heart Attack Risk: {percentage:.2f}%")

    # Interpretation
    if percentage < 30:
        st.success("✅ Low Risk")
    elif percentage < 70:
        st.warning("⚠️ Moderate Risk")
    else:
        st.error("🚨 High Risk")
