# Live Site Demo: https://falsefriend-nlp.streamlit.app/
# NLP False Friend Context Classification

This project implements a hybrid NLP classification system to detect the context of "False Friend" terms between Indonesian and Malaysian languages. It combines local Machine Learning with Large Language Model (LLM) benchmarking.

## Key Features
- **Data Reshaping:** Automatically processes wide-format datasets into labeled long-format samples (1,002 samples).
- **Optimized Local Model:** Random Forest Classifier fine-tuned using `GridSearchCV` with TF-IDF vectorization.
- **Advanced Hybrid Inference:** Real-time comparison between local model predictions and OpenRouter LLM (Llama 3.3/Free models).
- **Interactive Streaming UI:** Streamlit web application featuring live-typing responses and "Match/Conflict" logic.
- **Secure Configuration:** Environment variable management via `.env` for API security.

## Setup Instructions

1.  **Clone the repository & Setup Environment:**
    ```bash
    git clone <repository-url>
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    Create a `.env` file in the root directory and add your OpenRouter API Key:
    ```text
    OPENROUTER_API_KEY=your_api_key_here
    ```

3.  **Run Training:**
    Generate the serialized model files (`.pkl`) before running the app:
    ```bash
    python train.py
    ```

4.  **Launch Web App:**
    ```bash
    python -m streamlit run app.py
    ```

## Project Structure
- `data/`: Dataset source.
- `models/`: Serialized model artifacts.
- `notebooks/`: Academic research report and model implementation.
- `app.py`: Main Streamlit application.
- `train.py`: Training script for reproducibility.
- `REPORT_DRAFT.md`: Draft for academic report writing.
