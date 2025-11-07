"""
Train recommendation models (Content-Based, Collaborative Filtering)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import networkx as nx

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | <level>{message}</level>")

def train_content_based():
    """Train content-based recommendation model"""
    try:
        logger.info("📊 Training content-based recommender...")
        
        # Load data
        df = pd.read_csv("./data/processed/full_processed.csv")
        
        # Combine features for content similarity
        df['content_features'] = (
            df['title_clean'].fillna('') + ' ' +
            df['description_clean'].fillna('') + ' ' +
            df['listed_in'].fillna('') + ' ' +
            df['cast'].fillna('')
        )
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        content_matrix = tfidf.fit_transform(df['content_features'])
        
        logger.info(f"Content matrix shape: {content_matrix.shape}")
        
        # Save
        models_dir = Path("../backend/models/trained")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        with open(models_dir / "content_tfidf.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        
        with open(models_dir / "content_matrix.pkl", "wb") as f:
            pickle.dump(content_matrix, f)
        
        # Save content metadata
        metadata = df[['title', 'content_type', 'listed_in', 'rating']].to_dict('records')
        with open(models_dir / "content_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        logger.success("✅ Content-based model trained and saved!")
        
        return tfidf, content_matrix
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise

def train_collaborative_filtering():
    """Train collaborative filtering model (SVD)"""
    try:
        logger.info("🤝 Training collaborative filtering model...")
        
        # For demo, create synthetic ratings
        # In production, use real user ratings from Supabase
        logger.info("Creating synthetic user ratings for demo...")
        
        df = pd.read_csv("./data/processed/full_processed.csv")
        n_users = 100
        n_ratings_per_user = 20
        
        ratings_data = []
        for user_id in range(n_users):
            # Random items rated by user
            rated_items = np.random.choice(len(df), n_ratings_per_user, replace=False)
            for item_idx in rated_items:
                rating = np.random.uniform(1.0, 5.0)
                ratings_data.append({
                    'user_id': user_id,
                    'item_id': item_idx,
                    'rating': rating
                })
        
        ratings_df = pd.DataFrame(ratings_data)
        logger.info(f"Generated {len(ratings_df)} ratings")
        
        # Train SVD
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
        
        svd = SVD(n_factors=50, n_epochs=20, random_state=42)
        
        # Cross-validation
        logger.info("Running cross-validation...")
        cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
        
        logger.info(f"   RMSE: {cv_results['test_rmse'].mean():.4f}")
        logger.info(f"   MAE: {cv_results['test_mae'].mean():.4f}")
        
        # Train on full data
        trainset = data.build_full_trainset()
        svd.fit(trainset)
        
        # Save
        models_dir = Path("../backend/models/trained")
        with open(models_dir / "svd_model.pkl", "wb") as f:
            pickle.dump(svd, f)
        
        logger.success("✅ Collaborative filtering model trained and saved!")
        
        return svd
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise

def build_content_graph():
    """Build graph for graph-based recommendations"""
    try:
        logger.info("🕸️ Building content graph...")
        
        df = pd.read_csv("./data/processed/full_processed.csv")
        
        # Create graph
        G = nx.Graph()
        
        # Add content nodes
        for idx, row in df.iterrows():
            G.add_node(f"content_{idx}", type='content', title=row['title'])
        
        # Add edges based on shared genres/cast
        logger.info("Adding edges based on shared attributes...")
        
        # This is computationally expensive, so limit to a sample
        sample_size = min(500, len(df))
        df_sample = df.sample(sample_size)
        
        for i, row1 in df_sample.iterrows():
            genres1 = set(str(row1['listed_in']).split(','))
            for j, row2 in df_sample.iterrows():
                if i >= j:
                    continue
                genres2 = set(str(row2['listed_in']).split(','))
                overlap = len(genres1 & genres2)
                if overlap > 0:
                    G.add_edge(f"content_{i}", f"content_{j}", weight=overlap)
        
        logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Save graph
        models_dir = Path("../backend/models/trained")
        with open(models_dir / "content_graph.pkl", "wb") as f:
            pickle.dump(G, f)
        
        logger.success("✅ Content graph built and saved!")
        
        return G
        
    except Exception as e:
        logger.error(f"❌ Graph building failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Train content-based
    train_content_based()
    
    # Train collaborative filtering
    train_collaborative_filtering()
    
    # Build content graph
    build_content_graph()
