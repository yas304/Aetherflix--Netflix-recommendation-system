from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger
import sys

from api.routes import classify, recommend, user, health
from core.config import settings
from core.ml_loader import MLModelLoader
from db.supabase_client import supabase_client

# Configure logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")
logger.add("logs/app.log", rotation="500 MB", retention="10 days", compression="zip")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    logger.info("🚀 Starting AetherFlix AI Backend...")
    
    # Load ML models on startup
    try:
        ml_loader = MLModelLoader()
        await ml_loader.load_models()
        app.state.ml_loader = ml_loader
        logger.success("✅ ML models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load ML models: {e}")
        raise
    
    # Verify Supabase connection
    try:
        await supabase_client.verify_connection()
        logger.success("✅ Supabase connection verified")
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        raise
    
    logger.success("✨ AetherFlix AI Backend is ready!")
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down AetherFlix AI Backend...")
    await ml_loader.cleanup()
    logger.success("👋 Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="AetherFlix AI",
    description="Netflix-clone with AI-powered content classification and recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host Middleware
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(classify.router, prefix="/api", tags=["Classification"])
app.include_router(recommend.router, prefix="/api", tags=["Recommendations"])
app.include_router(user.router, prefix="/api", tags=["User"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to AetherFlix AI 🎬",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
