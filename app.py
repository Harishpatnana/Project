import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('catboost_model.pkl', 'rb'))

st.title("Heart Attack Risk Prediction ❤️")
st.write("Enter patient details")

# ===== REQUIRED FEATURES ONLY =====

thal = st.selectbox("Thalassemia (0–3)", [0,1,2,3])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
ca = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope (0–2)", [0,1,2])
restecg = st.selectbox("Rest ECG (0–2)", [0,1,2])
chol = st.number_input("Cholesterol", value=200)
thalach = st.number_input("Max Heart Rate", value=150)

# ===== PREDICTION =====

if st.button("Predict"):

    # EXACT order as training
    input_df = pd.DataFrame(
        [[thal, cp, ca, oldpeak, slope, restecg, chol, thalach]],
        columns=['thal', 'cp', 'ca', 'oldpeak', 'slope', 'restecg', 'chol', 'thalach']
    )

    # Probability
    prob = model.predict_proba(input_df)[0][1] * 100

    st.subheader(f"Heart Attack Risk: {prob:.2f}%")

    # Interpretation
    if prob < 30:
        st.success("✅ Low Risk")
    elif prob < 70:
        st.warning("⚠️ Moderate Risk")
    else:
        st.error("🚨 High Risk")
