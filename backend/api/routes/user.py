from fastapi import APIRouter, Depends, HTTPException, Header, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from loguru import logger
from typing import Optional

from api.schemas import UserProfile, UserRating, UserPreferences
from db.supabase_client import get_supabase

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/user/profile", response_model=UserProfile)
async def get_user_profile(
    authorization: Optional[str] = Header(None),
    supabase = Depends(get_supabase)
):
    """
    Get user profile information
    Requires JWT token in Authorization header
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(" ")[1]
    
    try:
        # Verify token with Supabase
        user = supabase.auth.get_user(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get user profile from database
        response = supabase.table("profiles").select("*").eq("id", user.user.id).execute()
        
        if not response.data:
            # Create default profile
            profile_data = {
                "id": user.user.id,
                "email": user.user.email,
                "name": user.user.user_metadata.get("name"),
                "avatar_url": user.user.user_metadata.get("avatar_url"),
                "preferences": {},
                "created_at": user.user.created_at
            }
            
            supabase.table("profiles").insert(profile_data).execute()
            return UserProfile(**profile_data)
        
        return UserProfile(**response.data[0])
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )

@router.post("/user/rating")
async def rate_content(
    rating: UserRating,
    authorization: Optional[str] = Header(None),
    supabase = Depends(get_supabase)
):
    """
    Submit user rating for content
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(" ")[1]
    
    try:
        user = supabase.auth.get_user(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Store rating
        rating_data = {
            "user_id": user.user.id,
            "content_id": rating.content_id,
            "rating": rating.rating,
            "timestamp": rating.timestamp
        }
        
        supabase.table("ratings").upsert(rating_data).execute()
        
        return {"message": "Rating saved successfully", "rating": rating.rating}
        
    except Exception as e:
        logger.error(f"Error saving rating: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save rating"
        )

@router.put("/user/preferences")
async def update_preferences(
    preferences: UserPreferences,
    authorization: Optional[str] = Header(None),
    supabase = Depends(get_supabase)
):
    """
    Update user preferences
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(" ")[1]
    
    try:
        user = supabase.auth.get_user(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Update preferences
        supabase.table("profiles").update({
            "preferences": preferences.dict()
        }).eq("id", user.user.id).execute()
        
        return {"message": "Preferences updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )
