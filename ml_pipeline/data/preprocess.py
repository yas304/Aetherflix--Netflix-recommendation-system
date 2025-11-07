"""
Preprocess Netflix dataset for ML training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import train_test_split
import re

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | <level>{message}</level>")

def clean_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip().lower()

def preprocess_dataset():
    """Main preprocessing pipeline"""
    try:
        logger.info("🔄 Starting data preprocessing...")
        
        # Load raw data
        raw_dir = Path("./data/raw")
        csv_files = list(raw_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in ./data/raw/")
        
        df = pd.read_csv(csv_files[0])
        logger.info(f"📊 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display column names
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Handle missing values
        logger.info("🧹 Cleaning data...")
        
        # Common Netflix dataset columns
        df = df.fillna({
            'director': 'Unknown',
            'cast': 'Unknown',
            'country': 'Unknown',
            'date_added': 'Unknown',
            'rating': 'NR',
            'duration': 'Unknown',
            'listed_in': 'Unknown',
            'description': ''
        })
        
        # Clean text columns
        if 'title' in df.columns:
            df['title_clean'] = df['title'].apply(clean_text)
        
        if 'description' in df.columns:
            df['description_clean'] = df['description'].apply(clean_text)
        
        # Process genres (listed_in)
        if 'listed_in' in df.columns:
            df['genres'] = df['listed_in'].str.split(',').apply(lambda x: [g.strip() for g in x] if isinstance(x, list) else ['Unknown'])
        
        # Extract year from date_added or release_year
        if 'release_year' in df.columns:
            df['year'] = pd.to_numeric(df['release_year'], errors='coerce')
        
        # Create content_type if not exists
        if 'type' in df.columns:
            df['content_type'] = df['type']
        
        # Feature engineering
        logger.info("⚙️ Engineering features...")
        
        # Text length features
        df['title_length'] = df['title'].str.len()
        df['desc_length'] = df['description'].str.len()
        
        # Cast size
        if 'cast' in df.columns:
            df['cast_size'] = df['cast'].str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Duration processing
        if 'duration' in df.columns:
            df['duration_value'] = df['duration'].str.extract(r'(\d+)').astype(float)
        
        # Split train/test
        logger.info("📊 Splitting train/test sets...")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Save processed data
        processed_dir = Path("./data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(processed_dir / "train.csv", index=False)
        test_df.to_csv(processed_dir / "test.csv", index=False)
        df.to_csv(processed_dir / "full_processed.csv", index=False)
        
        logger.success(f"✅ Preprocessing complete!")
        logger.info(f"   Train set: {len(train_df)} samples")
        logger.info(f"   Test set: {len(test_df)} samples")
        logger.info(f"   Saved to: {processed_dir}")
        
        # Display sample
        logger.info("\n📋 Sample data:")
        print(df[['title', 'content_type', 'listed_in']].head())
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    preprocess_dataset()
