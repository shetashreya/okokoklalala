from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    async def create_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Create embeddings for text(s)"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            # Convert to list format
            if len(texts) == 1:
                return embeddings[0].tolist()
            else:
                return [emb.tolist() for emb in embeddings]
                
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0