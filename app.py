import streamlit as st
import joblib

# ======================
# LOAD MODEL (PIPELINE)
# ======================
@st.cache_resource
def load_model():
    return joblib.load("model_svm.pkl")

model = load_model()

# ======================
# UI
# ======================
st.set_page_config(
    page_title="Mental Health Polarity Classification",
    layout="centered"
)

st.title("Mental Health Polarity Classification")
st.caption("Model: TF-IDF + Linear SVM (Pipeline)")

text_input = st.text_area(
    "Masukkan teks:",
    value="I feel calm and emotionally balanced today."
)

# ======================
# PREDICTION
# ======================
if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        pred = model.predict([text_input])[0]

        # ======================
        # LABEL MAPPING (BENAR)
        # ======================
        if pred == 0:
            label = "Mental Health Discourse / Potential Distress"
            st.error(f"Hasil: {label}")
        else:
            label = "Non Mental-Health Discourse"
            st.success(f"Hasil: {label}")

# ======================
# FOOTNOTE AKADEMIK
# ======================
st.markdown("---")
st.caption(
    "Catatan: Model ini mengklasifikasikan jenis *mental health discourse*, "
    "bukan analisis sentimen positif/negatif."
)
