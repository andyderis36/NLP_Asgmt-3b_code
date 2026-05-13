# NLP False Friend Context Classification

This project implements an NLP-based classification system to detect the context of "False Friend" terms between Indonesian and Malaysian languages.

## Features
- **Data Reshaping:** Automatically merges and cleans dual-language data.
- **Model Training:** Local Random Forest Classifier optimized with `GridSearchCV`.
- **Hybrid Inference:** Compares local model predictions with external LLM insights via OpenRouter.
- **Interactive UI:** Real-time context prediction via Streamlit.

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
    ```bash
    python train.py
    ```

4.  **Launch App:**
    ```bash
    python -m streamlit run app.py
    ```
