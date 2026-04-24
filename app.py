Kode
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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

model        = data["model"]
le_target    = data["le_target"]
cat_features = data["categorical_features"]
num_features = data["numerical_features"]

xgb_model    = model.named_steps["clf"]
preprocessor = model.named_steps["preprocess"]

# =============================
# LOAD EXPLAINER (sekali saja)
# =============================
@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(xgb_model)

explainer = load_explainer()

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
def skor_gaya_hidup(d):
    score = 0

    if d["FAF"] >= 2:
        score -= 1
    else:
        score += 1

    if d["FCVC"] >= 2:
        score -= 1
    else:
        score += 1

    if d["CH2O"] >= 2:
        score -= 1
    else:
        score += 1

    if d["FAVC"] == "yes":
        score += 1

    if d["CALC"] in ["Frequently", "Always"]:
        score += 1

    if d["family_history_with_overweight"] == "yes":
        score += 1

    if d["SCC"] == "yes":
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
def prediksi(d):
    df = pd.DataFrame([d])
    bmi = df["Weight"].iloc[0] / (df["Height"].iloc[0] ** 2)

    X = df[cat_features + num_features]
    X_transformed = preprocessor.transform(X)

    ohe       = preprocessor.named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(cat_features))
    all_names = ohe_names + num_features

    X_df = pd.DataFrame(X_transformed, columns=all_names)

    proba = xgb_model.predict_proba(X_df)[0]
    idx   = int(np.argmax(proba))

    return bmi, X_df, idx

# =============================
# FUNGSI TOP FAKTOR GAYA HIDUP
# =============================
def get_top_factors(X_df, predicted_class_idx, top_n=3):
    shap_vals = explainer.shap_values(X_df)

    if isinstance(shap_vals, list):
        sv = shap_vals[predicted_class_idx][0]
    else:
        sv = shap_vals[0, :, predicted_class_idx]

    # Hanya fitur gaya hidup — tanpa Weight, Height, Age, Gender
    lifestyle_features = [
        "FCVC", "NCP", "CH2O", "FAF", "TUE",
        "FAVC_yes", "FAVC_no",
        "CAEC_Sometimes", "CAEC_Frequently", "CAEC_Always", "CAEC_no",
        "SMOKE_yes", "SMOKE_no",
        "SCC_yes", "SCC_no",
        "CALC_Sometimes", "CALC_Frequently", "CALC_Always", "CALC_no",
        "MTRANS_Public_Transportation", "MTRANS_Walking",
        "MTRANS_Automobile", "MTRANS_Motorbike", "MTRANS_Bike",
        "family_history_with_overweight_yes",
        "family_history_with_overweight_no"
    ]

    label_map = {
        "FCVC"                               : "Konsumsi Sayur",
        "NCP"                                : "Jumlah Makan Utama",
        "CH2O"                               : "Konsumsi Air",
        "FAF"                                : "Aktivitas Fisik",
        "TUE"                                : "Penggunaan Teknologi",
        "FAVC_yes"                           : "Sering Konsumsi Makanan Tinggi Kalori",
        "FAVC_no"                            : "Jarang Konsumsi Makanan Tinggi Kalori",
        "CAEC_Sometimes"                     : "Kadang Ngemil",
        "CAEC_Frequently"                    : "Sering Ngemil",
        "CAEC_Always"                        : "Selalu Ngemil",
        "CAEC_no"                            : "Tidak Ngemil",
        "SMOKE_yes"                          : "Merokok",
        "SMOKE_no"                           : "Tidak Merokok",
        "SCC_yes"                            : "Monitoring Kalori",
        "SCC_no"                             : "Tidak Monitoring Kalori",
        "CALC_Sometimes"                     : "Kadang Konsumsi Alkohol",
        "CALC_Frequently"                    : "Sering Konsumsi Alkohol",
        "CALC_Always"                        : "Selalu Konsumsi Alkohol",
        "CALC_no"                            : "Tidak Konsumsi Alkohol",
        "MTRANS_Public_Transportation"       : "Transportasi Umum",
        "MTRANS_Walking"                     : "Jalan Kaki",
        "MTRANS_Automobile"                  : "Kendaraan Pribadi",
        "MTRANS_Motorbike"                   : "Motor",
        "MTRANS_Bike"                        : "Sepeda",
        "family_history_with_overweight_yes" : "Riwayat Keluarga Obesitas",
        "family_history_with_overweight_no"  : "Tidak Ada Riwayat Keluarga Obesitas",
    }

    feature_names = X_df.columns.tolist()
    shap_df = pd.DataFrame({
        "fitur" : feature_names,
        "nilai" : X_df.iloc[0].values,
        "shap"  : sv
    })

    # Filter hanya fitur gaya hidup
    shap_df = shap_df[shap_df["fitur"].isin(lifestyle_features)]
    shap_df["abs_shap"] = shap_df["shap"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).head(top_n)

    hasil = []
    for _, row in shap_df.iterrows():
        nama  = label_map.get(row["fitur"], row["fitur"])
        arah  = "Mendorong risiko" if row["shap"] > 0 else "Mengurangi risiko"
        emoji = "🔴" if row["shap"] > 0 else "🔵"
        hasil.append((emoji, nama, arah))

    return hasil

# =============================
# ANTARMUKA
# =============================
st.title("🏥 Sistem Penilaian Risiko Obesitas Klinis")
st.markdown("Evaluasi Risiko Obesitas Berbasis Machine Learning dan Faktor Gaya Hidup")

st.subheader("Input Data Pasien")

gender  = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age     = st.number_input("Usia (tahun)", 10, 80, 25)
height  = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, 1.70)
weight  = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0)

fcvc = st.slider("Konsumsi Sayur (1 = rendah, 3 = tinggi)", 1, 3, 2)
ncp  = st.slider("Jumlah Makan Utama per Hari", 1, 4, 3)
ch2o = st.slider("Konsumsi Air (1 = rendah, 3 = tinggi)", 1, 3, 2)
faf  = st.slider("Tingkat Aktivitas Fisik (0 = rendah, 3 = tinggi)", 0, 3, 1)
tue  = st.slider("Penggunaan Perangkat Teknologi (0-3)", 0, 3, 1)

family = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
favc   = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
smoke  = st.selectbox("Merokok", ["yes", "no"])
caec   = st.selectbox("Kebiasaan Ngemil", ["Sometimes", "Frequently", "Always", "no"])
calc   = st.selectbox("Konsumsi Alkohol", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Moda Transportasi Harian", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
scc    = st.selectbox("Melakukan Monitoring Kalori", ["yes", "no"])

if st.button("Analisis Risiko"):

    input_data = {
        "Gender"                        : gender,
        "family_history_with_overweight": family,
        "FAVC"                          : favc,
        "CAEC"                          : caec,
        "SMOKE"                         : smoke,
        "SCC"                           : scc,
        "CALC"                          : calc,
        "MTRANS"                        : mtrans,
        "Age"                           : age,
        "Height"                        : height,
        "Weight"                        : weight,
        "FCVC"                          : fcvc,
        "NCP"                           : ncp,
        "CH2O"                          : ch2o,
        "FAF"                           : faf,
        "TUE"                           : tue
    }

    bmi, X_df, predicted_idx = prediksi(input_data)
    lifestyle = skor_gaya_hidup(input_data)
    bmi_cat   = kategori_bmi(bmi)

    st.markdown("---")
    st.header("📋 Laporan Evaluasi Kesehatan")

    st.write(f"**Indeks Massa Tubuh (BMI):** {bmi:.2f}")
    st.write(f"**Kategori BMI:** {bmi_cat}")
    st.write(f"**Skor Risiko Gaya Hidup:** {lifestyle}")

    # ── Faktor Gaya Hidup dari SHAP ──
    st.markdown("### 🔍 Faktor Gaya Hidup yang Paling Berpengaruh")
    st.caption("Berdasarkan analisis model, berikut 3 faktor gaya hidup utama yang mempengaruhi hasil prediksi Anda:")

    with st.spinner("Menganalisis faktor risiko..."):
        try:
            top_factors = get_top_factors(X_df, predicted_idx, top_n=3)

            for i, (emoji, nama, arah) in enumerate(top_factors, 1):
                st.write(f"{i}. {emoji} **{nama}** → {arah}")

            st.caption("🔴 Mendorong risiko obesitas  |  🔵 Mengurangi risiko obesitas")

        except Exception as e:
            st.warning(f"Analisis tidak dapat ditampilkan: {e}")

    # ── Kesimpulan Klinis ──
    st.markdown("### 🧾 Kesimpulan Klinis")

    if bmi_cat == "Kelebihan Berat Badan (Overweight)" and lifestyle == "Rendah":
        st.info("Meskipun BMI menunjukkan kelebihan berat badan, pola gaya hidup tergolong baik. Risiko metabolik relatif rendah. Disarankan evaluasi komposisi tubuh lebih lanjut.")
    elif bmi_cat == "Kelebihan Berat Badan (Overweight)" and lifestyle != "Rendah":
        st.warning("BMI dan pola gaya hidup menunjukkan peningkatan risiko metabolik. Disarankan perbaikan pola makan dan peningkatan aktivitas fisik.")
    elif bmi_cat == "Obesitas":
        st.error("Risiko obesitas tinggi. Disarankan konsultasi medis dan intervensi gaya hidup segera.")
    else:
        st.success("Status berat badan dalam batas normal. Pertahankan gaya hidup sehat.")
