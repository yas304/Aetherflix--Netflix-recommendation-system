"""
Generate embeddings using Sentence-BERT and store in FAISS
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | <level>{message}</level>")

def generate_embeddings():
    """Generate embeddings for all content"""
    try:
        logger.info("🔮 Generating embeddings with Sentence-BERT...")
        
        # Load data
        df = pd.read_csv("./data/processed/full_processed.csv")
        logger.info(f"Processing {len(df)} items...")
        
        # Load model
        logger.info("Loading Sentence-BERT model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Prepare texts
        texts = []
        for _, row in df.iterrows():
            text = f"{row['title']} {row['description']} {row['listed_in']}"
            texts.append(text)
        
        # Generate embeddings
        logger.info("Generating embeddings (this may take a while)...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        
        # Save
        models_dir = Path("../backend/models/trained")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(models_dir / "faiss_index.index"))
        
        with open(models_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        
        # Save ID mapping
        id_mapping = df[['title', 'content_type', 'listed_in']].to_dict('records')
        with open(models_dir / "embedding_metadata.pkl", "wb") as f:
            pickle.dump(id_mapping, f)
        
        logger.success("✅ Embeddings generated and saved!")
        logger.info(f"   Saved to: {models_dir}")
        
        return embeddings, index
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    generate_embeddings()
