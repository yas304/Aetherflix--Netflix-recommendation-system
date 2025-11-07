# ML Training & Visualization Notebook
# Run this to train models and generate confusion matrices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
df = pd.read_csv('../backend/netflix_titles.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Data cleaning
df = df.dropna(subset=['type', 'description'])
df = df[df['type'].isin(['Movie', 'TV Show'])].reset_index(drop=True)
print(f"\nCleaned dataset: {len(df)} rows")

# Prepare features
X = df['description']
y = df['type']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# Create models directory
os.makedirs('../backend/models/trained', exist_ok=True)

# ========== LOGISTIC REGRESSION ==========
print("\n" + "="*50)
print("Training Logistic Regression...")
print("="*50)

lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=300, class_weight='balanced', random_state=RANDOM_SEED))
])

# Grid search
param_grid_lr = {
    'tfidf__max_features': [20000, 40000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.5, 1.0, 2.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
gs_lr = GridSearchCV(lr_pipeline, param_grid_lr, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
gs_lr.fit(X_train, y_train)

lr_best = gs_lr.best_estimator_
print(f"\nBest params: {gs_lr.best_params_}")

# Evaluate
y_pred_lr = lr_best.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"\n✅ Logistic Regression Accuracy: {acc_lr:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=['Movie', 'TV Show'])
plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Movie', 'TV Show'], 
            yticklabels=['Movie', 'TV Show'])
plt.title('Confusion Matrix - LogReg_Tuned')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../backend/models/trained/confusion_matrix_logreg.png', dpi=150)
print("✅ Saved: confusion_matrix_logreg.png")

# Save model
joblib.dump(lr_best, '../backend/models/trained/classifier_logreg.joblib')
print("✅ Saved: classifier_logreg.joblib")

# ========== LINEAR SVC ==========
print("\n" + "="*50)
print("Training Linear SVC...")
print("="*50)

svc_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2))),
    ('clf', LinearSVC(class_weight='balanced', random_state=RANDOM_SEED))
])

# Grid search
param_grid_svc = {
    'tfidf__max_features': [20000, 40000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.5, 1.0, 2.0]
}

gs_svc = GridSearchCV(svc_pipeline, param_grid_svc, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
gs_svc.fit(X_train, y_train)

svc_best = gs_svc.best_estimator_
print(f"\nBest params: {gs_svc.best_params_}")

# Evaluate
y_pred_svc = svc_best.predict(X_test)
acc_svc = accuracy_score(y_test, y_pred_svc)
print(f"\n✅ Linear SVC Accuracy: {acc_svc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svc))

# Confusion Matrix
cm_svc = confusion_matrix(y_test, y_pred_svc, labels=['Movie', 'TV Show'])
plt.figure(figsize=(6,5))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Movie', 'TV Show'],
            yticklabels=['Movie', 'TV Show'])
plt.title('Confusion Matrix - LinearSVC_Tuned')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../backend/models/trained/confusion_matrix_linearsvc.png', dpi=150)
print("✅ Saved: confusion_matrix_linearsvc.png")

# Save model
joblib.dump(svc_best, '../backend/models/trained/classifier_svc.joblib')
print("✅ Saved: classifier_svc.joblib")

# ========== DATASET VISUALIZATIONS ==========
print("\n" + "="*50)
print("Creating dataset visualizations...")
print("="*50)

# 1. Movie vs TV Show count
plt.figure(figsize=(7,5))
type_counts = df['type'].value_counts()
plt.bar(type_counts.index, type_counts.values, color=['#E50914', '#564d4d'])
plt.title('Count: Movie vs TV Show', fontsize=14, fontweight='bold')
plt.xlabel('type')
plt.ylabel('count')
plt.tight_layout()
plt.savefig('../backend/models/trained/count_movie_vs_tvshow.png', dpi=150)
print("✅ Saved: count_movie_vs_tvshow.png")

# 2. Titles by Release Year
plt.figure(figsize=(10,5))
year_counts = df['release_year'].value_counts().sort_index()
plt.plot(year_counts.index, year_counts.values, linewidth=2, color='#0080FF')
plt.title('Titles by Release Year', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../backend/models/trained/titles_by_year.png', dpi=150)
print("✅ Saved: titles_by_year.png")

# ========== SAVE METADATA ==========
metadata = {
    'logreg_accuracy': float(acc_lr),
    'svc_accuracy': float(acc_svc),
    'best_model': 'svc' if acc_svc > acc_lr else 'logreg',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'total_movies': int((df['type'] == 'Movie').sum()),
    'total_tvshows': int((df['type'] == 'TV Show').sum())
}

import json
with open('../backend/models/trained/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: metadata.json")

print("\n" + "="*50)
print("✨ TRAINING COMPLETE!")
print("="*50)
print(f"\nLogistic Regression: {acc_lr:.2%}")
print(f"Linear SVC: {acc_svc:.2%}")
print(f"\n🏆 Best Model: {metadata['best_model'].upper()} ({max(acc_lr, acc_svc):.2%})")
