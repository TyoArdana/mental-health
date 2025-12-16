import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load("model_svm.pkl")

model = load_model()

st.set_page_config(
    page_title="Mental Health Polarity Classification",
    layout="centered"
)

st.title("Mental Health Polarity Classification")
st.write("**Model:** TF-IDF + Linear SVM (Pipeline)")

text_input = st.text_area(
    "Masukkan teks:",
    placeholder="Contoh: I feel stressed and anxious lately."
)

if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        pred = model.predict([text_input])[0]

        if pred == 1:
            st.success("Hasil: Positive / Healthy")
        else:
            st.error("Hasil: Negative / Poor")
