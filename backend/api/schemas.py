from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ContentType(str, Enum):
    MOVIE = "Movie"
    TV_SHOW = "TV Show"

class Genre(str, Enum):
    ACTION = "Action"
    COMEDY = "Comedy"
    DRAMA = "Drama"
    HORROR = "Horror"
    THRILLER = "Thriller"
    ROMANCE = "Romance"
    SCIFI = "Sci-Fi"
    FANTASY = "Fantasy"
    DOCUMENTARY = "Documentary"
    ANIMATION = "Animation"

# Classification Models
class ClassificationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=10, max_length=5000)
    poster_url: Optional[str] = None
    cast: Optional[List[str]] = None
    director: Optional[str] = None
    release_year: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Stranger Things",
                "description": "A group of kids in a small town face supernatural forces and secret government experiments",
                "poster_url": "https://image.tmdb.org/t/p/w500/poster.jpg",
                "cast": ["Millie Bobby Brown", "Finn Wolfhard"],
                "director": "The Duffer Brothers",
                "release_year": 2016
            }
        }

class ClassificationResponse(BaseModel):
    content_type: ContentType
    genres: List[str]
    confidence_scores: Dict[str, float]
    rating: Optional[str] = None
    model_version: str = "1.0.0"
    
# Recommendation Models
class RecommendationRequest(BaseModel):
    user_id: Optional[str] = None
    content_id: Optional[str] = None
    query: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)
    method: str = Field(default="hybrid", pattern="^(content|collaborative|hybrid|semantic)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "query": "Thrillers like Stranger Things",
                "limit": 10,
                "method": "hybrid"
            }
        }

class ContentItem(BaseModel):
    id: str
    title: str
    description: str
    poster_url: Optional[str] = None
    genres: List[str]
    content_type: ContentType
    rating: Optional[float] = None
    release_year: Optional[int] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)

class RecommendationResponse(BaseModel):
    recommendations: List[ContentItem]
    method_used: str
    total_count: int
    user_id: Optional[str] = None

# User Models
class UserProfile(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    created_at: datetime
    
class UserRating(BaseModel):
    user_id: str
    content_id: str
    rating: float = Field(..., ge=0.0, le=5.0)
    timestamp: Optional[datetime] = None

class UserPreferences(BaseModel):
    favorite_genres: List[str] = []
    disliked_genres: List[str] = []
    preferred_content_type: Optional[ContentType] = None
    
# Health Check Models
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]
