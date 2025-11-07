"""
Complete ML Training Pipeline for AetherFlix AI
Trains classification and recommendation models from Netflix dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load Netflix dataset
print("📊 Loading Netflix dataset...")
df = pd.read_csv('../backend/netflix_titles.csv')

print(f"✅ Loaded {len(df)} titles")
print(f"Movies: {len(df[df['type']=='Movie'])}, TV Shows: {len(df[df['type']=='TV Show'])}")

# Preprocessing
print("\n🔧 Preprocessing data...")
df['description'] = df['description'].fillna('')
df['listed_in'] = df['listed_in'].fillna('')
df['cast'] = df['cast'].fillna('')
df['director'] = df['director'].fillna('')

# Create combined features for classification
df['combined_features'] = df['description'] + ' ' + df['listed_in'] + ' ' + df['cast']

# PART 1: TYPE CLASSIFICATION (Movie vs TV Show)
print("\n🤖 Training Type Classification Model...")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['combined_features'])
y = df['type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LogisticRegression
print("   Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"   ✅ Logistic Regression Accuracy: {lr_acc:.3f}")

# Train LinearSVC
print("   Training Linear SVC...")
svc_model = LinearSVC(max_iter=1000, random_state=42)
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)

print(f"   ✅ Linear SVC Accuracy: {svc_acc:.3f}")

# Generate Confusion Matrices
print("\n📊 Generating confusion matrices...")

# Create visualizations directory
import os
os.makedirs('../frontend/public/ml_charts', exist_ok=True)

# Confusion Matrix - LogReg
plt.figure(figsize=(8, 6))
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Movie', 'TV Show'],
            yticklabels=['Movie', 'TV Show'])
plt.title('Confusion Matrix - LogReg_Tuned')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../frontend/public/ml_charts/confusion_matrix_logreg.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved confusion_matrix_logreg.png")
plt.close()

# Confusion Matrix - LinearSVC
plt.figure(figsize=(8, 6))
cm_svc = confusion_matrix(y_test, svc_pred)
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Movie', 'TV Show'],
            yticklabels=['Movie', 'TV Show'])
plt.title('Confusion Matrix - LinearSVC_Tuned')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../frontend/public/ml_charts/confusion_matrix_linearsvc.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved confusion_matrix_linearsvc.png")
plt.close()

# Titles by Release Year
print("\n📈 Generating release year chart...")
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
year_counts = df.groupby('release_year').size().reset_index(name='count')
year_counts = year_counts[year_counts['release_year'] >= 1920]

plt.figure(figsize=(12, 6))
plt.plot(year_counts['release_year'], year_counts['count'], linewidth=2)
plt.title('Titles by Release Year', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../frontend/public/ml_charts/titles_by_year.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved titles_by_year.png")
plt.close()

# Movie vs TV Show Count
print("📊 Generating type distribution chart...")
type_counts = df['type'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(type_counts.index, type_counts.values, color='steelblue')
plt.title('Count: Movie vs TV Show', fontsize=14, fontweight='bold')
plt.xlabel('type')
plt.ylabel('count')
plt.tight_layout()
plt.savefig('../frontend/public/ml_charts/type_distribution.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved type_distribution.png")
plt.close()

# PART 2: RECOMMENDATION ENGINE
print("\n🎯 Building Recommendation Engine...")

# TF-IDF for recommendations
tfidf_rec = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_rec.fit_transform(df['combined_features'])

# Compute cosine similarity
print("   Computing similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim, df=df, top_n=10):
    """Get top N recommendations for a given title"""
    idx = df[df['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return []
    
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Skip first (itself)
    
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations = df.iloc[movie_indices][['title', 'type', 'listed_in', 'description']].to_dict('records')
    return recommendations

# Test recommendations
print("\n🧪 Testing recommendation engine...")
test_title = df.iloc[0]['title']
recs = get_recommendations(test_title, top_n=5)
print(f"   Recommendations for '{test_title}':")
for i, rec in enumerate(recs[:3], 1):
    print(f"   {i}. {rec['title']}")

# Save models
print("\n💾 Saving models...")
os.makedirs('../backend/models/trained', exist_ok=True)

# Save classification models
with open('../backend/models/trained/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("   ✅ Saved tfidf_vectorizer.pkl")

with open('../backend/models/trained/logreg_classifier.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("   ✅ Saved logreg_classifier.pkl")

with open('../backend/models/trained/svc_classifier.pkl', 'wb') as f:
    pickle.dump(svc_model, f)
print("   ✅ Saved svc_classifier.pkl")

# Save recommendation models
with open('../backend/models/trained/tfidf_recommender.pkl', 'wb') as f:
    pickle.dump(tfidf_rec, f)
print("   ✅ Saved tfidf_recommender.pkl")

with open('../backend/models/trained/cosine_similarity.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)
print("   ✅ Saved cosine_similarity.pkl")

# Save processed dataset
df_export = df[['show_id', 'type', 'title', 'director', 'cast', 'country', 
                'date_added', 'release_year', 'rating', 'duration', 
                'listed_in', 'description']].copy()
df_export.to_csv('../backend/processed_netflix_data.csv', index=False)
print("   ✅ Saved processed_netflix_data.csv")

# Generate summary stats
print("\n📊 Model Summary:")
print("="*50)
print(f"Dataset Size: {len(df)} titles")
print(f"Movies: {len(df[df['type']=='Movie'])}")
print(f"TV Shows: {len(df[df['type']=='TV Show'])}")
print(f"\nClassification Accuracy:")
print(f"  - Logistic Regression: {lr_acc*100:.2f}%")
print(f"  - Linear SVC: {svc_acc*100:.2f}%")
print(f"\nRecommendation Engine:")
print(f"  - TF-IDF Features: 5000")
print(f"  - Similarity Metric: Cosine")
print("="*50)

print("\n✨ All models trained and saved successfully!")
print("🎉 Ready to power AetherFlix AI!")
