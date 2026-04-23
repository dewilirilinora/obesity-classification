import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
data = joblib.load("model_obesitas.joblib")

model        = data["model"]           # pipeline (preprocess + clf)
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

    # Kolom sesuai urutan pipeline
    X = df[cat_features + num_features]

    # Transform pakai preprocessor dari pipeline
    X_transformed = preprocessor.transform(X)

    # Ambil nama fitur hasil OHE
    ohe       = preprocessor.named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(cat_features))
    all_names = ohe_names + num_features

    # Buat DataFrame untuk SHAP
    X_df = pd.DataFrame(X_transformed, columns=all_names)

    # Prediksi
    proba = xgb_model.predict_proba(X_df)[0]
    idx   = int(np.argmax(proba))

    return bmi, X_df, idx

# =============================
# FUNGSI SHAP PLOT
# =============================
def get_shap_plot(X_df, predicted_class_idx):
    shap_vals = explainer.shap_values(X_df)

    exp = shap.Explanation(
        values        = shap_vals[predicted_class_idx][0],
        base_values   = explainer.expected_value[predicted_class_idx],
        data          = X_df.iloc[0].values,
        feature_names = X_df.columns.tolist()
    )

    fig, _ = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    return fig

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

    # ── SHAP ──
    st.markdown("### 🔍 Analisis Faktor Risiko (SHAP)")
    st.caption("Grafik berikut menunjukkan faktor-faktor yang mempengaruhi prediksi model")

    with st.spinner("Menghitung analisis SHAP..."):
        try:
            fig = get_shap_plot(X_df, predicted_idx)
            st.pyplot(fig)
            plt.close()

            st.caption("""
            📌 **Cara membaca grafik:**
            - 🔴 **Merah (+)** = Faktor yang mendorong ke arah prediksi ini
            - 🔵 **Biru (-)** = Faktor yang mengurangi risiko prediksi ini
            - Semakin panjang bar = semakin besar pengaruhnya
            """)
        except Exception as e:
            st.warning(f"SHAP tidak dapat ditampilkan: {e}")

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
