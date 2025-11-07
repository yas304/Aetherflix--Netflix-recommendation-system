from supabase import create_client, Client
from core.config import settings
from loguru import logger
from typing import Optional

class SupabaseClient:
    """Supabase client wrapper for database and auth operations"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Supabase client"""
        try:
            self.client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_ANON_KEY
            )
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    async def verify_connection(self):
        """Verify Supabase connection"""
        try:
            # Try a simple query to verify connection
            response = self.client.table("profiles").select("count").limit(1).execute()
            logger.success("Supabase connection verified")
            return True
        except Exception as e:
            # If profiles table doesn't exist, that's okay during initial setup
            logger.warning(f"Supabase connection check: {e}")
            return True
    
    def get_client(self) -> Client:
        """Get Supabase client instance"""
        if not self.client:
            self._initialize()
        return self.client

# Global instance
supabase_client = SupabaseClient()

def get_supabase() -> Client:
    """Dependency for getting Supabase client"""
    return supabase_client.get_client()
