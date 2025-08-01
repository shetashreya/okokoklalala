import aiofiles
import asyncio
import tempfile
import os
from typing import List, Dict, Any
import PyPDF2
from docx import Document
import httpx
from app.core.config import settings
from app.models.schemas import DocumentChunk
import hashlib
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    async def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise Exception(f"Failed to download document: {str(e)}")
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                
                text = ""
                with open(tmp_file.name, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                
                os.unlink(tmp_file.name)
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise Exception(f"Failed to extract PDF text: {str(e)}")
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                
                doc = Document(tmp_file.name)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                
                os.unlink(tmp_file.name)
                return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise Exception(f"Failed to extract DOCX text: {str(e)}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into chunks with overlap"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Create unique ID for chunk
            chunk_id = hashlib.md5(
                (chunk_text + str(i)).encode()
            ).hexdigest()
            
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "start_word": i,
                "end_word": min(i + self.chunk_size, len(words)),
                "word_count": len(chunk_words)
            }
            
            chunks.append(DocumentChunk(
                id=chunk_id,
                content=chunk_text.strip(),
                metadata=chunk_metadata
            ))
        
        return chunks
    
    async def process_document(self, document_url: str) -> List[DocumentChunk]:
        """Process document from URL and return chunks"""
        try:
            # Download document
            logger.info(f"Downloading document from: {document_url}")
            content = await self.download_document(document_url)
            
            # Determine file type and extract text
            if document_url.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(content)
                doc_type = "pdf"
            elif document_url.lower().endswith('.docx'):
                text = self.extract_text_from_docx(content)
                doc_type = "docx"
            else:
                # Try PDF first, then DOCX
                try:
                    text = self.extract_text_from_pdf(content)
                    doc_type = "pdf"
                except:
                    text = self.extract_text_from_docx(content)
                    doc_type = "docx"
            
            # Create metadata
            metadata = {
                "source_url": document_url,
                "document_type": doc_type,
                "character_count": len(text),
                "processing_timestamp": asyncio.get_event_loop().time()
            }
            
            # Chunk the text
            chunks = self.chunk_text(text, metadata)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise Exception(f"Failed to process document: {str(e)}")