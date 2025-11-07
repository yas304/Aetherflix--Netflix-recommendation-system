from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path
from collections import Counter

router = APIRouter()
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "trained"

try:
    with open(MODEL_PATH / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(MODEL_PATH / "logreg_classifier.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open(MODEL_PATH / "svc_classifier.pkl", "rb") as f:
        svc_model = pickle.load(f)
    df = pd.read_csv(Path(__file__).parent.parent.parent / "processed_netflix_data.csv")
    print(" Classification models loaded")
except Exception as e:
    print(f" Model error: {e}")
    tfidf = lr_model = svc_model = df = None

class ClassifyRequest(BaseModel):
    text: str

@router.post("/classify")
async def classify_content(request: ClassifyRequest):
    if not all([tfidf, lr_model, df is not None]):
        raise HTTPException(503, "Models not loaded")
    
    try:
        X = tfidf.transform([request.text])
        prediction = lr_model.predict(X)[0]
        proba = lr_model.predict_proba(X)[0]
        confidence = max(proba)
        
        predicted_df = df[df["type"] == prediction].head(100)
        all_genres = " ".join(predicted_df["listed_in"].fillna("").values)
        genre_list = [g.strip() for g in all_genres.split(",")]
        top_genres = [item[0] for item in Counter(genre_list).most_common(5) if item[0]]
        
        return {
            "predicted_type": prediction,
            "confidence": float(confidence),
            "suggested_genres": top_genres
        }
    except Exception as e:
        raise HTTPException(500, str(e))
