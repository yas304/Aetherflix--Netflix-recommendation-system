from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from datetime import datetime

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "operational",
            "ml_models": "operational",
            "database": "operational"
        }
    }

@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong"}
