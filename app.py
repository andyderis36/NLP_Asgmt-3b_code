import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

st.title("Deteksi Konteks False Friend (Indo vs Malay)")

user_input = st.text_input("Masukkan kalimat:")

if st.button("Deteksi"):
    if user_input:
        tfidf_vec = vectorizer.transform([user_input])
        prediction = model.predict(tfidf_vec)[0]
        st.success(f"Konteks: {prediction}")
    else:
        st.warning("Masukkan kalimat terlebih dahulu.")
