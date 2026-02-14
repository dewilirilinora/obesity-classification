import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Clinical Obesity Risk Assessment",
    page_icon="🏥",
    layout="centered"
)

# =============================
# LOAD MODEL
# =============================
with open("model_obesitas.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"] if "model" in data else data["model2"]
scaler = data["scaler"]
ohe = data["ohe"]
le_target = data["le_target"]
numerical_features = data["numerical_features"]
binary_cat = data["binary_cat"]
multi_cat = data["multi_cat"]

# =============================
# BMI CATEGORY
# =============================
def kategori_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# =============================
# LIFESTYLE RISK SCORE
# =============================
def lifestyle_score(data):
    score = 0
    
    # Aktivitas fisik (protective)
    if data["FAF"] >= 2:
        score -= 1
    else:
        score += 1

    # Konsumsi sayur
    if data["FCVC"] >= 2:
        score -= 1
    else:
        score += 1

    # Air minum
    if data["CH2O"] >= 2:
        score -= 1
    else:
        score += 1

    # Makanan tinggi kalori
    if data["FAVC"] == "yes":
        score += 1

    # Alkohol
    if data["CALC"] in ["Frequently", "Always"]:
        score += 1

    # Riwayat keluarga
    if data["family_history_with_overweight"] == "yes":
        score += 1

    # Monitoring kalori
    if data["SCC"] == "yes":
        score -= 1

    if score <= -1:
        return "Rendah"
    elif score <= 2:
        return "Sedang"
    else:
        return "Tinggi"

# =============================
# PREDICTION FUNCTION
# =============================
def predict(data):
    df = pd.DataFrame([data])
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    X_num = df[numerical_features].values
    X_binary = np.array([
        1 if df[col].values[0] in ["yes", "Male"] else 0
        for col in binary_cat
    ]).reshape(1, -1)
    X_multi = ohe.transform(df[multi_cat])

    X_final = np.hstack([X_num, X_binary, X_multi])
    X_scaled = scaler.transform(X_final)

    proba = model.predict_proba(X_scaled)[0]
    idx = np.argmax(proba)
    diagnosis = le_target.inverse_transform([idx])[0]

    return diagnosis, proba[idx], df["BMI"].iloc[0]

# =============================
# UI
# =============================
st.title("🏥 Clinical Obesity Risk Assessment")
st.markdown("Sistem Evaluasi Risiko Obesitas Berbasis Machine Learning")

st.subheader("Data Pasien")

gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Usia", 10, 80, 25)
height = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, 1.70)
weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0)

fcvc = st.slider("Konsumsi Sayur (1-3)", 1, 3, 2)
ncp = st.slider("Makan Utama per Hari (1-4)", 1, 4, 3)
ch2o = st.slider("Konsumsi Air (1-3)", 1, 3, 2)
faf = st.slider("Aktivitas Fisik (0-3)", 0, 3, 1)
tue = st.slider("Penggunaan Teknologi (0-3)", 0, 3, 1)

family = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
favc = st.selectbox("Sering Makan Tinggi Kalori", ["yes", "no"])
caec = st.selectbox("Ngemil", ["Sometimes", "Frequently", "Always", "no"])
calc = st.selectbox("Konsumsi Alkohol", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
scc = st.selectbox("Monitoring Kalori", ["yes", "no"])

if st.button("Analisis Risiko"):

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

    diagnosis, confidence, bmi = predict(input_data)
    lifestyle = lifestyle_score(input_data)
    bmi_cat = kategori_bmi(bmi)

    st.markdown("---")
    st.header("📋 Laporan Evaluasi Kesehatan")

    st.write(f"**BMI:** {bmi:.2f}")
    st.write(f"**Kategori BMI:** {bmi_cat}")
    st.write(f"**Skor Risiko Gaya Hidup:** {lifestyle}")
    st.write(f"**Prediksi Machine Learning:** {diagnosis}")
    st.write(f"**Probabilitas:** {confidence:.2%}")

    st.markdown("### 🧾 Penilaian Terintegrasi")

    if bmi_cat == "Overweight" and lifestyle == "Rendah":
        st.info("Meskipun BMI menunjukkan overweight, pola gaya hidup relatif sehat. Kemungkinan komposisi tubuh dipengaruhi massa otot. Disarankan evaluasi komposisi tubuh lanjutan.")
    elif bmi_cat == "Overweight" and lifestyle != "Rendah":
        st.warning("BMI dan faktor gaya hidup menunjukkan peningkatan risiko metabolik. Disarankan intervensi pola makan dan aktivitas fisik.")
    elif bmi_cat == "Obese":
        st.error("Risiko obesitas tinggi. Disarankan konsultasi medis dan intervensi segera.")
    else:
        st.success("Status berat badan dalam batas normal. Pertahankan gaya hidup sehat.")

    st.markdown("---")
    st.caption("Developed for Undergraduate Thesis - 2026")
