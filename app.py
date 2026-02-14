import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================
# LOAD MODEL
# =============================
with open("model_obesitas.pkl", "rb") as f:
    data = pickle.load(f)

model1 = data["model1"]
model2 = data["model2"]
model3 = data["model3"]
scaler = data["scaler"]
ohe = data["ohe"]
le_target = data["le_target"]
numerical_features = data["numerical_features"]
binary_cat = data["binary_cat"]
multi_cat = data["multi_cat"]

# =============================
# PREDICTION FUNCTION
# =============================
def predict_obesity(input_data):
    df = pd.DataFrame([input_data])
    
    # BMI auto calculation
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # Numerical
    X_num = df[numerical_features].values

    # Binary
    X_binary = np.array([
        1 if df[col].values[0] in ["yes", "Male"] else 0
        for col in binary_cat
    ]).reshape(1, -1)

    # Multi-cat
    X_multi = ohe.transform(df[multi_cat])

    # Combine
    X_final = np.hstack([X_num, X_binary, X_multi])
    X_scaled = scaler.transform(X_final)

    # Ensemble
    proba1 = model1.predict_proba(X_scaled)[0]
    proba2 = model2.predict_proba(X_scaled)[0]
    proba3 = model3.predict_proba(X_scaled)[0]

    final_proba = 0.3 * proba1 + 0.5 * proba2 + 0.2 * proba3

    best_idx = np.argmax(final_proba)
    diagnosis = le_target.inverse_transform([best_idx])[0]
    confidence = final_proba[best_idx]

    return diagnosis, confidence, round(df["BMI"].iloc[0], 2)

# =============================
# UI STREAMLIT
# =============================
st.title("OBESITY LEVEL PREDICTION SYSTEM")
st.write("Machine Learning Based Obesity Classification")

st.subheader("Patient Information")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 10, 80, 21)
height = st.number_input("Height (meters)", 1.0, 2.5, 1.70)
weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)

fcvc = st.slider("Vegetable Consumption (FCVC)", 1, 3, 2)
ncp = st.slider("Main Meals per Day (NCP)", 1, 4, 3)
ch2o = st.slider("Water Intake (CH2O)", 1, 3, 2)
faf = st.slider("Physical Activity (FAF)", 0, 3, 1)
tue = st.slider("Technology Usage (TUE)", 0, 3, 1)

family = st.selectbox("Family History Overweight", ["yes", "no"])
favc = st.selectbox("High Calorie Food", ["yes", "no"])
caec = st.selectbox("Snack Between Meals", ["Sometimes", "Frequently", "Always", "no"])
calc = st.selectbox("Alcohol Consumption", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Transportation", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
scc = st.selectbox("Calories Monitoring", ["yes", "no"])

if st.button("Predict"):
    
    input_data = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue,
        "family_history_with_overweight": family,
        "FAVC": favc,
        "CAEC": caec,
        "CALC": calc,
        "MTRANS": mtrans,
        "SCC": scc
    }

    diagnosis, confidence, bmi = predict_obesity(input_data)

    st.success(f"Predicted Class: {diagnosis}")
    st.info(f"Confidence: {confidence:.2%}")
    st.warning(f"Calculated BMI: {bmi}")
