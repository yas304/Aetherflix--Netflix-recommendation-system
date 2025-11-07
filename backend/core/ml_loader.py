import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import pickle
import onnxruntime as ort
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

class MLModelLoader:
    """Load and manage ML models for inference"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.faiss_index: Optional[faiss.Index] = None
        self.model_path = Path("./models/trained")
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    async def load_models(self):
        """Load all ML models asynchronously"""
        logger.info("Loading ML models...")
        
        # Load classification models
        await self._load_classifier()
        
        # Load recommendation models
        await self._load_recommender()
        
        # Load FAISS index
        await self._load_vector_db()
        
        # Load sentence transformer
        await self._load_sentence_transformer()
        
        logger.success("All ML models loaded successfully")
    
    async def _load_classifier(self):
        """Load classification model (ONNX format for speed)"""
        try:
            classifier_path = self.model_path / "classifier.onnx"
            if classifier_path.exists():
                self.models["classifier"] = ort.InferenceSession(
                    str(classifier_path),
                    providers=['CPUExecutionProvider']
                )
                logger.info("✅ Classifier model loaded (ONNX)")
            else:
                logger.warning("⚠️ Classifier model not found, will use fallback")
                # Fallback to XGBoost pickle
                xgb_path = self.model_path / "xgboost_classifier.pkl"
                if xgb_path.exists():
                    with open(xgb_path, "rb") as f:
                        self.models["classifier"] = pickle.load(f)
                    logger.info("✅ XGBoost classifier loaded (pickle)")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            self.models["classifier"] = None
    
    async def _load_recommender(self):
        """Load recommendation models"""
        try:
            # TF-IDF vectorizer
            tfidf_path = self.model_path / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                with open(tfidf_path, "rb") as f:
                    self.models["tfidf"] = pickle.load(f)
                logger.info("✅ TF-IDF vectorizer loaded")
            
            # SVD model for collaborative filtering
            svd_path = self.model_path / "svd_model.pkl"
            if svd_path.exists():
                with open(svd_path, "rb") as f:
                    self.models["svd"] = pickle.load(f)
                logger.info("✅ SVD model loaded")
            
            # Content matrix
            content_matrix_path = self.model_path / "content_matrix.pkl"
            if content_matrix_path.exists():
                with open(content_matrix_path, "rb") as f:
                    self.models["content_matrix"] = pickle.load(f)
                logger.info("✅ Content matrix loaded")
        except Exception as e:
            logger.error(f"Failed to load recommender models: {e}")
    
    async def _load_vector_db(self):
        """Load FAISS vector index"""
        try:
            faiss_path = self.model_path / "faiss_index.index"
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info(f"✅ FAISS index loaded ({self.faiss_index.ntotal} vectors)")
            else:
                logger.warning("⚠️ FAISS index not found")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
    
    async def _load_sentence_transformer(self):
        """Load sentence transformer for semantic embeddings"""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Check if model exists locally
            local_path = self.model_path / "sentence_transformer"
            if local_path.exists():
                self.tokenizers["sentence"] = AutoTokenizer.from_pretrained(str(local_path))
                self.models["sentence_transformer"] = AutoModel.from_pretrained(str(local_path))
                logger.info("✅ Sentence transformer loaded (local)")
            else:
                # Download from Hugging Face
                self.tokenizers["sentence"] = AutoTokenizer.from_pretrained(model_name)
                self.models["sentence_transformer"] = AutoModel.from_pretrained(model_name)
                logger.info("✅ Sentence transformer loaded (HuggingFace)")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.models["sentence_transformer"] = None
    
    def get_model(self, model_name: str):
        """Get loaded model by name"""
        return self.models.get(model_name)
    
    def get_tokenizer(self, tokenizer_name: str):
        """Get loaded tokenizer by name"""
        return self.tokenizers.get(tokenizer_name)
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up ML models...")
        self.models.clear()
        self.tokenizers.clear()
        self.faiss_index = None
        logger.success("Cleanup complete")
