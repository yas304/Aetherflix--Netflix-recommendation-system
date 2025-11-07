from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path

router = APIRouter()
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "trained"

try:
    with open(MODEL_PATH / "cosine_similarity.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    df = pd.read_csv(Path(__file__).parent.parent.parent / "processed_netflix_data.csv")
    print(" Recommendation engine loaded")
except Exception as e:
    print(f" Recommendation error: {e}")
    cosine_sim = df = None

class RecommendRequest(BaseModel):
    title: str
    top_n: int = 10

@router.post("/recommend")
async def get_recommendations(request: RecommendRequest):
    if cosine_sim is None or df is None:
        raise HTTPException(503, "Engine not loaded")
    
    try:
        idx = df[df["title"].str.lower() == request.title.lower()].index
        if len(idx) == 0:
            raise HTTPException(404, f"Title not found")
        
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:request.top_n+1]
        
        indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[indices][["title", "type", "listed_in", "description", "release_year", "rating"]].to_dict("records")
        
        return {"recommendations": recommendations}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/browse")
async def browse_content(type: str = None, limit: int = 50):
    if df is None:
        raise HTTPException(503, "Dataset not loaded")
    
    try:
        filtered = df.copy()
        if type:
            filtered = filtered[filtered["type"] == type]
        
        results = filtered.head(limit)[["title", "type", "listed_in", "description", "release_year", "rating"]].to_dict("records")
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))
