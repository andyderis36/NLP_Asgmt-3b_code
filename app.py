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
    # Prompt dioptimalkan agar model memberikan format yang lebih mudah diproses
    prompt = f"""Analisis kalimat ini: '{text}'. 
    Tentukan apakah kalimat ini konteksnya Bahasa Indonesia atau Bahasa Melayu (Malaysia).
    JAWAB HANYA DENGAN SATU KATA: 'Indonesia' atau 'Malaysia'.
    Setelah itu, berikan penjelasan singkatnya.
    Format:
    Label: [Indonesia/Malaysia]
    Alasan: [penjelasan singkat]"""

    try:
        stream = client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
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
        return "Error"

st.title("Deteksi Konteks False Friend (Indo vs Malay)")

user_input = st.text_input("Masukkan kalimat:")

if st.button("Deteksi"):
    if user_input:
        # Prediction Lokal
        tfidf_vec = vectorizer.transform([user_input])
        local_pred = model.predict(tfidf_vec)[0]

        # UI Comparison
        col1, col2 = st.columns(2)
        col1.subheader("Random Forest (Lokal)")
        col1.write(f"**{local_pred}**")

        col2.subheader("OpenRouter (Eksternal)")
        placeholder = col2.empty()

        # Prediction OpenRouter
        with st.spinner('Meminta bantuan OpenRouter...'):
            try:
                stream = client.chat.completions.create(
                    model="openai/gpt-oss-120b:free",
                    messages=[{"role": "user", "content": f"Analisis kalimat ini: '{user_input}'. Tentukan apakah kalimat ini menggunakan konteks Bahasa Indonesia atau Bahasa Melayu (Malaysia) berdasarkan penggunaan kata 'False Friend' di dalamnya. Jawab dengan format: Label: [Indonesia/Malaysia] diikuti dengan Alasan: [penjelasan singkat]."}],
                    stream=True
                )
                
                openrouter_resp = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        openrouter_resp += chunk.choices[0].delta.content
                        placeholder.markdown(openrouter_resp + "▌")
                
                placeholder.markdown(openrouter_resp)
                
                # Parsing yang lebih presisi
                if "Label: Indonesia" in openrouter_resp or "Indonesia" in openrouter_resp.split("Label:")[1].split()[0]:
                    openrouter_pred = "Indonesia"
                else:
                    openrouter_pred = "Malaysia"
            except Exception as e:
                st.error(f"Error OpenRouter: {e}")
                openrouter_pred = "Error"
                openrouter_resp = "Gagal mendapatkan respons."

            # Match Indicator dengan normalisasi (lowercase & strip)
            if local_pred.strip().lower() == openrouter_pred.strip().lower():
                st.success(f"Result: Match! (Local: {local_pred}, OpenRouter: {openrouter_pred})")
            else:
                st.warning(f"Result: Conflict! (Local: {local_pred}, OpenRouter: {openrouter_pred})")
