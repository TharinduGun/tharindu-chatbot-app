from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.services.rag import RAGPipeline
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    provider: Optional[str] = "openai"  # openai, groq, ollama

class ChatResponse(BaseModel):
    answer: str
    retrieval_stats: Dict[str, int]
    sources: List[str]

@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Submits a query to the RAG pipeline.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        # Initialize RAG Pipeline (Models are cached singletons, so this is cheap)
        pipeline = RAGPipeline(llm_provider=request.provider)
        
        # Run
        result = await pipeline.answer_query(request.query)
        
        return ChatResponse(
            answer=result["answer"],
            retrieval_stats=result["retrieval_stats"],
            sources=list(set(result["sources"])) # Dedup
        )
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
