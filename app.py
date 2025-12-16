import streamlit as st
import joblib

model = joblib.load("model_svm.pkl")

st.title("Mental Health Polarity Classification")
st.write("Model: TF-IDF + Linear SVM")

text = st.text_area("Masukkan teks:")

pred = model.predict(X_input)[0]

if pred == 0:
    label = "Mental Health Discourse / Potential Distress"
    st.error(f"Hasil: {label}")
else:
    label = "Neutral / No Distress Indicators"
    st.success(f"Hasil: {label}")

