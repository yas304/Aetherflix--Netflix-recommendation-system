"""
Train classification models (XGBoost, BERT)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | <level>{message}</level>")

def train_xgboost_classifier():
    """Train XGBoost classifier for content type and genres"""
    try:
        logger.info("🎯 Training XGBoost classifier...")
        
        # Load data
        train_df = pd.read_csv("./data/processed/train.csv")
        test_df = pd.read_csv("./data/processed/test.csv")
        
        # Prepare features
        logger.info("Preparing features...")
        
        # Combine text features
        train_df['text'] = train_df['title_clean'].fillna('') + ' ' + train_df['description_clean'].fillna('')
        test_df['text'] = test_df['title_clean'].fillna('') + ' ' + test_df['description_clean'].fillna('')
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = tfidf.fit_transform(train_df['text'])
        X_test = tfidf.transform(test_df['text'])
        
        # Target: content_type
        y_train = (train_df['content_type'] == 'Movie').astype(int)
        y_test = (test_df['content_type'] == 'Movie').astype(int)
        
        # Train XGBoost
        logger.info("Training XGBoost model...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.success(f"✅ XGBoost trained successfully!")
        logger.info(f"   F1 Score: {f1:.4f}")
        logger.info(f"   ROC-AUC: {roc_auc:.4f}")
        
        # Save models
        models_dir = Path("../backend/models/trained")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        with open(models_dir / "xgboost_classifier.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open(models_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        
        logger.success(f"💾 Models saved to {models_dir}")
        
        return model, tfidf
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise

def train_bert_classifier():
    """Train BERT fine-tuned classifier (optional, resource-intensive)"""
    logger.info("🤖 BERT training skipped (requires GPU and time)")
    logger.info("💡 Use XGBoost model for now, fine-tune BERT separately if needed")

if __name__ == "__main__":
    # Train XGBoost (fast, works on CPU)
    train_xgboost_classifier()
    
    # BERT training commented out (requires GPU)
    # train_bert_classifier()
