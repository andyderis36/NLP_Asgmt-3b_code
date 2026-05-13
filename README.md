# NLP False Friend Context Classification

This project implements an NLP-based classification system to detect the context of "False Friend" terms (words that look similar but have different meanings) between Indonesian and Malaysian languages.

## Project Structure

```text
├── data/               # Dataset (CSV)
├── models/             # Exported model files (.pkl)
├── notebooks/          # Exploratory Data Analysis & Prototyping
├── app.py              # Streamlit web application
├── train.py            # Model training and export script
├── requirements.txt    # Project dependencies
└── .gitignore          # Git exclusion rules
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <project-folder>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model
To train the model and generate the serialized model files (`.pkl`), run the training script:
```bash
python train.py
```

### 2. Running the Application
Launch the interactive Streamlit web interface:
```bash
streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your browser to interact with the model.

## Features
- **Data Reshaping:** Automatically merges and cleans dual-language data.
- **Model Tuning:** Uses `GridSearchCV` to optimize the Random Forest Classifier.
- **Interactive UI:** Real-time context prediction via Streamlit.
