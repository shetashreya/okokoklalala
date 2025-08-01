from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse
from app.services.retrieval_service import RetrievalService
from app.core.security import verify_token
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()
retrieval_service = RetrievalService()

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """
    Main endpoint for processing documents and answering questions
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Validate input
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document URL is required"
            )
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )
        
        # Process the query batch
        answers = await retrieval_service.process_query_batch(
            document_url=request.documents,
            questions=request.questions
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Request processed successfully in {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing request after {processing_time:.2f} seconds: {e}")
        
        # Return error responses for all questions
        error_answers = [
            f"An error occurred while processing this question: {str(e)}"
            for _ in request.questions
        ]
        
        return QueryResponse(answers=error_answers)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        system_status = await retrieval_service.get_system_status()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system": system_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@router.get("/system/status")
async def get_system_status(token: str = Depends(verify_token)):
    """Get detailed system status (requires authentication)"""
    try:
        status_info = await retrieval_service.get_system_status()
        return status_info
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )