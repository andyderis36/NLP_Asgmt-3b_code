import streamlit as st
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Setup OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Load model dan vectorizer
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def get_llm_prediction(text, placeholder):
    prompt = f"Analisis kalimat ini: '{text}'. Tentukan apakah kalimat ini menggunakan konteks Bahasa Indonesia atau Bahasa Melayu (Malaysia) berdasarkan penggunaan kata 'False Friend' di dalamnya. Jawab dengan format: Label: [Indonesia/Malaysia] diikuti dengan Alasan: [penjelasan singkat]."

    try:
        stream = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        return full_response
    except Exception as e:
        placeholder.error(f"Maaf, terjadi kesalahan saat menghubungi server model: {e}")
        return "Gagal mendapatkan respons."

st.title("Deteksi Konteks False Friend (Indo vs Malay)")

user_input = st.text_input("Masukkan kalimat:")

if st.button("Deteksi"):
    if user_input:
        # Prediction Lokal
        tfidf_vec = vectorizer.transform([user_input])
        local_pred = model.predict(tfidf_vec)[0]
        
        # UI Comparison
        col1, col2 = st.columns(2)
        col1.subheader("Random Forest (Model Lokal)")
        col1.write(f"**{local_pred}**")
        
        col2.subheader("OpenRouter (Model Eksternal)")
        placeholder = col2.empty()
        
        # Prediction OpenRouter
        with st.spinner('Meminta bantuan OpenRouter...'):
            openrouter_resp = get_llm_prediction(user_input, placeholder)
            # Simple parsing for comparison
            openrouter_pred = "Indonesia" if "Indonesia" in openrouter_resp else "Malaysia"
            
        # Match Indicator
        if local_pred == openrouter_pred:
            st.success("Result: Match!")
        else:
            st.warning("Result: Conflict!")
    else:
        st.warning("Masukkan kalimat terlebih dahulu.")
