import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

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
# LOAD SHAP EXPLAINER
# =============================
@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(xgb_model)

explainer = load_explainer()

# =============================
# BMI
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
# SKOR GAYA HIDUP (RULE-BASED)
# =============================
def skor_gaya_hidup(d):
    score = 0

    score += -1 if d["FAF"] >= 2 else 1
    score += -1 if d["FCVC"] >= 2 else 1
    score += -1 if d["CH2O"] >= 2 else 1

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
# PREDIKSI
# =============================
def prediksi(d):
    df = pd.DataFrame([d])

    bmi = df["Weight"].iloc[0] / (df["Height"].iloc[0] ** 2)

    X = df[cat_features + num_features]
    X_transformed = preprocessor.transform(X)

    ohe = preprocessor.named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(cat_features))
    all_names = ohe_names + num_features

    X_df = pd.DataFrame(X_transformed, columns=all_names)

    proba = xgb_model.predict_proba(X_df)[0]
    idx   = int(np.argmax(proba))

    return bmi, X_df, idx

# =============================
# SHAP - TOP LIFESTYLE FACTORS
# =============================
def get_top_factors(X_df, predicted_class_idx, top_n=3):

    shap_vals = explainer(X_df)
    sv = shap_vals.values[0, :, predicted_class_idx]

    shap_df = pd.DataFrame({
        "fitur": X_df.columns,
        "shap": sv
    })

    # 🔥 SORT GLOBAL DULU
    shap_df["abs_shap"] = shap_df["shap"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False)

    # 🔥 FILTER LIFESTYLE (SETELAH SORT)
    lifestyle_keywords = [
        "FCVC","NCP","CH2O","FAF","TUE",
        "FAVC","CAEC","SMOKE","SCC","CALC","MTRANS"
    ]

    shap_df = shap_df[
        shap_df["fitur"].apply(lambda x: any(k in x for k in lifestyle_keywords))
    ]

    shap_df = shap_df.head(top_n)

    # 🔥 mapping manusiawi
    label_map = {
        "FCVC": "Konsumsi Sayur",
        "NCP": "Jumlah Makan",
        "CH2O": "Konsumsi Air",
        "FAF": "Aktivitas Fisik",
        "TUE": "Penggunaan Teknologi",
        "FAVC": "Makanan Tinggi Kalori",
        "CAEC": "Kebiasaan Ngemil",
        "SMOKE": "Merokok",
        "SCC": "Monitoring Kalori",
        "CALC": "Konsumsi Alkohol",
        "MTRANS": "Transportasi"
    }

    hasil = []

    for _, row in shap_df.iterrows():
        nama_raw = row["fitur"]

        nama = next(
            (v for k, v in label_map.items() if k in nama_raw),
            nama_raw
        )

        arah  = "Mendorong risiko" if row["shap"] > 0 else "Mengurangi risiko"
        emoji = "🔴" if row["shap"] > 0 else "🔵"

        hasil.append((emoji, nama, arah))

    return hasil

# =============================
# UI
# =============================
st.title("🏥 Sistem Penilaian Risiko Obesitas Klinis")
st.markdown("Berbasis Machine Learning + Interpretasi SHAP")

st.subheader("Input Data")

gender  = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age     = st.number_input("Usia", 10, 80, 25)
height  = st.number_input("Tinggi (m)", 1.0, 2.5, 1.70)
weight  = st.number_input("Berat (kg)", 30.0, 200.0, 70.0)

fcvc = st.slider("Konsumsi Sayur", 1, 3, 2)
ncp  = st.slider("Jumlah Makan", 1, 4, 3)
ch2o = st.slider("Air", 1, 3, 2)
faf  = st.slider("Aktivitas Fisik", 0, 3, 1)
tue  = st.slider("Teknologi", 0, 3, 1)

family = st.selectbox("Riwayat Keluarga", ["yes", "no"])
favc   = st.selectbox("Makanan Tinggi Kalori", ["yes", "no"])
smoke  = st.selectbox("Merokok", ["yes", "no"])
caec   = st.selectbox("Ngemil", ["Sometimes", "Frequently", "Always", "no"])
calc   = st.selectbox("Alkohol", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
scc    = st.selectbox("Monitoring Kalori", ["yes", "no"])

# =============================
# ANALISIS
# =============================
if st.button("Analisis"):

    input_data = {
        "Gender": gender,
        "family_history_with_overweight": family,
        "FAVC": favc,
        "CAEC": caec,
        "SMOKE": smoke,
        "SCC": scc,
        "CALC": calc,
        "MTRANS": mtrans,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue
    }

    bmi, X_df, pred_idx = prediksi(input_data)
    lifestyle = skor_gaya_hidup(input_data)

    st.markdown("---")
    st.header("📋 Hasil Evaluasi")

    st.write(f"**BMI:** {bmi:.2f}")
    st.write(f"**Kategori:** {kategori_bmi(bmi)}")
    st.write(f"**Risiko Lifestyle:** {lifestyle}")

    st.markdown("### 🔍 Faktor Gaya Hidup Utama")

    top_factors = get_top_factors(X_df, pred_idx)

    for i, (emoji, nama, arah) in enumerate(top_factors, 1):
        st.write(f"{i}. {emoji} {nama} → {arah}")
