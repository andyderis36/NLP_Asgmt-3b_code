# Project Report: Hybrid Classification System for Malay–Indonesian False Friends

## 1. Introduction
This research explores "False Friends"—lexical items shared between Bahasa Melayu (BM) and Bahasa Indonesia (BI) that possess divergent meanings. The goal is to evaluate cross-lingual semantic understanding using a hybrid approach combining local supervised learning and external Large Language Models (LLMs).

## 2. Methodology
### 2.1 Data Pipeline
- **Reshaping:** We transformed the wide-format dataset (501 pairs) into a long-format corpus (1,002 labeled samples) to ensure class balance and lexical diversity.
- **Preprocessing:** Handling null values and normalizing text input.

### 2.2 Implementation & Feature Engineering
- **Feature Representation:** TF-IDF (Term Frequency-Inverse Document Frequency) was selected to capture unique lexical markers crucial for disambiguating False Friends (Cahyawijaya et al., 2025).
- **Modeling:** 
    - **Baseline:** Multinomial Naive Bayes.
    - **Proposed:** Random Forest Classifier, optimized via `GridSearchCV` for hyperparameter tuning (n_estimators, max_depth).

### 2.3 Hybrid Inference System
- Integration of **OpenRouter API** to provide external benchmarking.
- Implementation of **Streaming Responses** for an interactive user experience.
- **Comparison Logic:** A robust "Match/Conflict" detection system that normalizes predictions (lowercase/strip) to validate local model performance against LLM reasoning.

## 3. Experimental Setup
- **Train/Test Split:** 80/20 ratio.
- **Metrics:** Accuracy and F1-Score visualized through styled comparative tables and Seaborn Confusion Matrices.
- **Reproducibility:** Environment managed via `venv` and `.env`, with all dependencies documented in `requirements.txt`.

## 4. Discussion & Conclusion
The results demonstrate that while the local Random Forest model is effective at capturing dataset-specific patterns, the LLM provides broader linguistic context. The "Conflict" flags serve as a valuable diagnostic tool for identifying edge cases where semantic overlap between BM and BI is particularly high.

## 5. Implementation Details
- **Tools:** Python, Scikit-learn, Pandas, Streamlit, OpenAI SDK.
- **Infrastructure:** Local training with cloud-based benchmarking via OpenRouter.
