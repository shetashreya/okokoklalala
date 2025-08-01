import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_KEY: str = "c61acf6dfe00a39f662ac0e4c9dbebf0700f169710c2e07dd95e56636418ab65"
    HOST: str = "localhost"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Grok API Configuration
    GROK_API_KEY: str
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    GROK_MODEL: str = "grok-beta"
    
    # Qdrant Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    COLLECTION_NAME: str = "hackrx_documents"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 4000
    
    class Config:
        env_file = ".env"

settings = Settings()