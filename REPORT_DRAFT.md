# Project Report: False Friend Context Classification System

## 1. Introduction
This project focuses on the linguistic challenges of "False Friends"—words that look similar but possess different meanings—between Indonesian and Malaysian. The objective is to build a robust classification system that accurately detects the context of such terms.

## 2. Methodology
### 2.1 Data Processing
- Dataset: `Dataset Group E - False Friend.csv`
- Reshaping: The dataset was converted from wide format to a unified long format consisting of text samples and their corresponding language labels (Indonesia/Malaysia).
- Cleaning: Null values were removed to ensure model stability.

### 2.2 Model Implementation
- **Baseline:** Multinomial Naive Bayes using TF-IDF vectorization.
- **Advanced:** Random Forest Classifier optimized with `GridSearchCV` (Hyperparameters: `n_estimators`, `max_depth`).
- **Hybrid Inference:** Integration of OpenRouter API (Llama 3.3/OpenRouter free model) for comparative analysis against the local model.

## 3. Experimental Results
- Local model performance was evaluated using Accuracy and F1-Score.
- A confusion matrix was implemented using Seaborn to visualize classification performance.
- A "Match/Conflict" logic was introduced to compare the local model's performance against the LLM's reasoning capabilities.

## 4. Discussion & Conclusion
- The hybrid approach provides a mechanism for human-in-the-loop validation, where model conflicts serve as data points for future model fine-tuning.
- The application was deployed via Streamlit, providing an interactive, user-friendly interface.

## 5. Future Work
- Expand the dataset to include more regional variations.
- Implement fine-tuning for the Random Forest model using samples flagged as "Conflict".
