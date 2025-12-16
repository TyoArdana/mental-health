import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Mental Health Classification", layout="centered")

st.title("Mental Health Polarity Classification")
st.caption("Model: TF-IDF + Linear SVM")

# Input teks
text = st.text_area("Masukkan teks:", height=150)

# Tombol prediksi
if st.button("Prediksi"):

    if text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        # === WAJIB ADA SEBELUM predict ===
        X_input = vectorizer.transform([text])

        pred = model.predict(X_input)[0]

        if pred == 0:
            st.error("Hasil: Mental Health Discourse / Potential Distress")
        else:
            st.success("Hasil: Neutral / No Distress Indicators")
