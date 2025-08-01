import httpx
import json
from typing import List, Dict, Any
from app.core.config import settings
from app.models.schemas import RetrievalResult
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = settings.GROK_API_KEY
        self.base_url = settings.GROK_BASE_URL
        self.model = settings.GROK_MODEL
        self.max_tokens = settings.MAX_TOKENS
    
    async def generate_answer(self, question: str, context_chunks: List[RetrievalResult]) -> str:
        """Generate answer using Grok LLM with retrieved context"""
        try:
            # Prepare context from retrieved chunks
            context = self._prepare_context(context_chunks)
            
            # Create the prompt
            prompt = self._create_prompt(question, context)
            
            # Call Grok API
            answer = await self._call_grok_api(prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I couldn't generate an answer for this question due to a technical error: {str(e)}"
    
    def _prepare_context(self, context_chunks: List[RetrievalResult]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        
        for i, result in enumerate(context_chunks[:5]):  # Use top 5 chunks
            chunk_text = result.chunk.content
            score = result.score
            
            context_parts.append(f"""
Context {i+1} (Relevance: {score:.3f}):
{chunk_text}
---
""")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a detailed prompt for the LLM"""
        prompt = f"""You are an expert document analyst specializing in insurance, legal, HR, and compliance domains. Your task is to answer questions based ONLY on the provided context from official documents.

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context
2. Be precise and specific with details like numbers, dates, percentages, and conditions
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Provide explanations and rationale when applicable
5. Structure your answer clearly and professionally
6. Include relevant conditions, limitations, or exceptions mentioned in the document

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

ANSWER: Based on the provided document context, """
        
        return prompt
    
    async def _call_grok_api(self, prompt: str) -> str:
        """Make API call to Grok"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1,  # Low temperature for factual responses
                "top_p": 0.9
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0]["message"]["content"].strip()
                    return answer
                else:
                    raise Exception("No response generated from LLM")
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Grok API: {e}")
            raise Exception(f"API request failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error calling Grok API: {e}")
            raise Exception(f"Failed to get LLM response: {str(e)}")
    
    async def extract_key_phrases(self, question: str) -> List[str]:
        """Extract key phrases from question for better retrieval"""
        try:
            prompt = f"""Extract the most important keywords and phrases from this question that would be useful for searching in insurance/legal documents. Return only the key terms, separated by commas.

Question: {question}

Key terms:"""
            
            response = await self._call_grok_api(prompt)
            
            # Parse the response to get key phrases
            phrases = [phrase.strip() for phrase in response.split(',')]
            return [phrase for phrase in phrases if phrase and len(phrase) > 2]
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            # Return simple fallback
            return [word for word in question.split() if len(word) > 3]