import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Sistem Penilaian Risiko Obesitas Klinis",
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
# FUNGSI KATEGORI BMI
# =============================
def kategori_bmi(bmi):
    if bmi < 18.5:
        return "Berat Badan Kurang"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Kelebihan Berat Badan (Overweight)"
    else:
        return "Obesitas"

# =============================
# FUNGSI SKOR RISIKO GAYA HIDUP
# =============================
def skor_gaya_hidup(data):
    score = 0

    if data["FAF"] >= 2:
        score -= 1
    else:
        score += 1

    if data["FCVC"] >= 2:
        score -= 1
    else:
        score += 1

    if data["CH2O"] >= 2:
        score -= 1
    else:
        score += 1

    if data["FAVC"] == "yes":
        score += 1

    if data["CALC"] in ["Frequently", "Always"]:
        score += 1

    if data["family_history_with_overweight"] == "yes":
        score += 1

    if data["SCC"] == "yes":
        score -= 1

    if score <= -1:
        return "Rendah"
    elif score <= 2:
        return "Sedang"
    else:
        return "Tinggi"

# =============================
# FUNGSI PREDIKSI
# =============================
def prediksi(data):
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

    # Rapikan nama kelas
    diagnosis = diagnosis.replace("_", " ")

    return diagnosis, proba[idx], df["BMI"].iloc[0]

# =============================
# ANTARMUKA
# =============================
st.title("🏥 Sistem Penilaian Risiko Obesitas Klinis")
st.markdown("Evaluasi Risiko Obesitas Berbasis Machine Learning dan Faktor Gaya Hidup")

st.subheader("Input Data Pasien")

gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Usia (tahun)", 10, 80, 25)
height = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, 1.70)
weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0)

fcvc = st.slider("Konsumsi Sayur (1 = rendah, 3 = tinggi)", 1, 3, 2)
ncp = st.slider("Jumlah Makan Utama per Hari", 1, 4, 3)
ch2o = st.slider("Konsumsi Air (1 = rendah, 3 = tinggi)", 1, 3, 2)
faf = st.slider("Tingkat Aktivitas Fisik (0 = rendah, 3 = tinggi)", 0, 3, 1)
tue = st.slider("Penggunaan Perangkat Teknologi (0-3)", 0, 3, 1)

family = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
caec = st.selectbox("Kebiasaan Ngemil", ["Sometimes", "Frequently", "Always", "no"])
calc = st.selectbox("Konsumsi Alkohol", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Moda Transportasi Harian", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
scc = st.selectbox("Melakukan Monitoring Kalori", ["yes", "no"])

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

    diagnosis, confidence, bmi = prediksi(input_data)
    lifestyle = skor_gaya_hidup(input_data)
    bmi_cat = kategori_bmi(bmi)

    st.markdown("---")
    st.header("📋 Laporan Evaluasi Kesehatan")

    st.write(f"**Indeks Massa Tubuh (BMI):** {bmi:.2f}")
    st.write(f"**Kategori BMI:** {bmi_cat}")
    st.write(f"**Skor Risiko Gaya Hidup:** {lifestyle}")
    st.write(f"**Prediksi Model Machine Learning:** {diagnosis}")
    st.write(f"**Tingkat Keyakinan Model:** {confidence:.2%}")

    st.markdown("### 🧾 Kesimpulan Klinis")

    if bmi_cat == "Kelebihan Berat Badan (Overweight)" and lifestyle == "Rendah":
        st.info("Meskipun BMI menunjukkan kelebihan berat badan, pola gaya hidup tergolong baik. Risiko metabolik relatif rendah. Disarankan evaluasi komposisi tubuh lebih lanjut.")
    elif bmi_cat == "Kelebihan Berat Badan (Overweight)" and lifestyle != "Rendah":
        st.warning("BMI dan pola gaya hidup menunjukkan peningkatan risiko metabolik. Disarankan perbaikan pola makan dan peningkatan aktivitas fisik.")
    elif bmi_cat == "Obesitas":
        st.error("Risiko obesitas tinggi. Disarankan konsultasi medis dan intervensi gaya hidup segera.")
    else:
        st.success("Status berat badan dalam batas normal. Pertahankan gaya hidup sehat.")
