import re
import logging
from typing import List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    """Validate if the provided string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length while preserving word boundaries"""
    if len(text) <= max_length:
        return text
    
    # Find the last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can preserve most of the text
        return text[:last_space] + "..."
    else:
        return text[:max_length] + "..."

def extract_document_type(url: str) -> str:
    """Extract document type from URL"""
    url_lower = url.lower()
    
    if '.pdf' in url_lower:
        return 'pdf'
    elif '.docx' in url_lower:
        return 'docx'
    elif '.doc' in url_lower:
        return 'doc'
    else:
        return 'unknown'

def format_error_response(error: Exception, context: str = "") -> Dict[str, Any]:
    """Format error response for consistent error handling"""
    return {
        "error": type(error).__name__,
        "message": str(error),
        "context": context
    }

def calculate_token_estimate(text: str) -> int:
    """Rough estimate of token count for text"""
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
    
    if not filename:
        filename = "document"
    
    return filename

def get_file_extension(url: str) -> str:
    """Extract file extension from URL"""
    try:
        path = urlparse(url).path
        return path.split('.')[-1].lower() if '.' in path else ''
    except Exception:
        return ''

def validate_questions(questions: List[str]) -> List[str]:
    """Validate and clean questions list"""
    validated = []
    
    for question in questions:
        if isinstance(question, str) and question.strip():
            cleaned = clean_text(question.strip())
            if cleaned and len(cleaned) > 5:  # Minimum question length
                validated.append(cleaned)
    
    return validated

def log_processing_stats(stats: Dict[str, Any]):
    """Log processing statistics"""
    logger.info("Processing Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

def format_processing_time(seconds: float) -> str:
    """Format processing time in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"