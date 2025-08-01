from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float
    relevance: str

class ProcessingStatus(BaseModel):
    status: str
    message: str
    processed_chunks: int
    total_chunks: int

class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int