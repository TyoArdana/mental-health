import streamlit as st
import joblib

model = joblib.load("model_svm.pkl")

st.title("Mental Health Polarity Classification")
st.write("Model: TF-IDF + Linear SVM")

text = st.text_area("Masukkan teks:")

if st.button("Prediksi"):
    if text.strip():
        pred = model.predict([text])[0]
        label = "Positive / Fair" if pred == 1 else "Negative / Poor"
        st.success(f"Hasil: {label}")
    else:
        st.warning("Teks tidak boleh kosong.")
