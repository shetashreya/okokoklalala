from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.models.schemas import DocumentChunk, RetrievalResult
import logging
import hashlib

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        self.client = None
        self.collection_name = settings.COLLECTION_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            if settings.QDRANT_API_KEY:
                self.client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY
                )
            else:
                self.client = QdrantClient(url=settings.QDRANT_URL)
            
            logger.info("Qdrant client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise Exception(f"Failed to initialize Qdrant client: {str(e)}")
    
    async def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise Exception(f"Failed to create collection: {str(e)}")
    
    async def store_chunks(self, chunks: List[DocumentChunk]):
        """Store document chunks with embeddings in Qdrant"""
        try:
            await self.create_collection()
            
            points = []
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                    continue
                
                point = PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload={
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Stored {len(points)} chunks in Qdrant")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise Exception(f"Failed to store chunks: {str(e)}")
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        score_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """Search for similar chunks using vector similarity"""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            results = []
            for point in search_result:
                chunk = DocumentChunk(
                    id=str(point.id),
                    content=point.payload["content"],
                    metadata=point.payload["metadata"],
                    embedding=None  # Don't return embeddings to save memory
                )
                
                result = RetrievalResult(
                    chunk=chunk,
                    score=point.score,
                    relevance=self._get_relevance_label(point.score)
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            raise Exception(f"Failed to search chunks: {str(e)}")
    
    def _get_relevance_label(self, score: float) -> str:
        """Convert score to relevance label"""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    async def clear_collection(self):
        """Clear all points from the collection"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise Exception(f"Failed to clear collection: {str(e)}")