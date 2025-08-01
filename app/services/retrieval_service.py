from typing import List, Dict, Any
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.simple_vector_store import create_vector_store
from app.core.config import settings
from app.models.schemas import DocumentChunk, RetrievalResult
import logging
import asyncio

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.vector_db = create_vector_store(use_qdrant=settings.USE_QDRANT)
    
    async def process_and_store_document(self, document_url: str) -> Dict[str, Any]:
        """Process document and store in vector database"""
        try:
            logger.info(f"Processing document: {document_url}")
            
            # Process document into chunks
            chunks = await self.document_processor.process_document(document_url)
            
            # Create embeddings for all chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.create_embeddings(chunk_texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Store in vector database
            await self.vector_db.store_chunks(chunks)
            
            return {
                "status": "success",
                "chunks_processed": len(chunks),
                "document_url": document_url
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise Exception(f"Failed to process document: {str(e)}")
    
    async def answer_question(self, question: str) -> str:
        """Answer a question using retrieval-augmented generation"""
        try:
            logger.info(f"Answering question: {question}")
            
            # Create embedding for the question
            question_embedding = await self.embedding_service.create_embeddings(question)
            
            # Search for relevant chunks
            relevant_chunks = await self.vector_db.search_similar_chunks(
                query_embedding=question_embedding,
                limit=10,
                score_threshold=0.3
            )
            
            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer this question."
            
            # Generate answer using LLM with retrieved context
            answer = await self.llm_service.generate_answer(question, relevant_chunks)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    async def process_query_batch(self, document_url: str, questions: List[str]) -> List[str]:
        """Process a batch of questions for a document"""
        try:
            # First, process and store the document
            await self.process_and_store_document(document_url)
            
            # Then answer all questions
            answers = []
            for question in questions:
                try:
                    answer = await self.answer_question(question)
                    answers.append(answer)
                except Exception as e:
                    logger.error(f"Error answering question '{question}': {e}")
                    answers.append(f"Unable to answer this question due to an error: {str(e)}")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error processing query batch: {e}")
            # Return error messages for all questions
            return [f"Error processing document: {str(e)}" for _ in questions]
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        try:
            collection_info = await self.vector_db.get_collection_info()
            
            return {
                "status": "operational",
                "vector_database": collection_info,
                "embedding_model": self.embedding_service.model_name,
                "llm_model": self.llm_service.model
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }