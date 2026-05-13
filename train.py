import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# 1. Load & Reshape Data
df = pd.read_csv('data/Dataset Group E - False Friend.csv')
df_indo = df[['Sentence_Clean_Indo']].rename(columns={'Sentence_Clean_Indo': 'text'})
df_indo['label'] = 'Indonesia'
df_malay = df[['Sentence_Clean_Malay']].rename(columns={'Sentence_Clean_Malay': 'text'})
df_malay['label'] = 'Malaysia'
df_final = pd.concat([df_indo, df_malay], ignore_index=True).dropna(subset=['text'])

# 2. Vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df_final['text'])
y = df_final['label']

# 3. Training & Tuning
print("Melatih model (ini mungkin memakan waktu)...")
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_tfidf, y)

# 4. Save Model & Vectorizer
joblib.dump(grid_search.best_estimator_, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model dan Vectorizer berhasil disimpan di folder 'models/'.")
