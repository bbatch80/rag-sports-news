"""
FastAPI backend for RAG Sports News (Production Hardened).

Security features:
- API key authentication
- Rate limiting (prevents OpenAI cost abuse)
- Restricted CORS origins
- Sanitized error messages
"""

import os
from functools import wraps

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from query import query, Answer
from retriever import search, SearchResult
from embeddings import get_collection_stats

load_dotenv()

# --- Configuration ---

# API key for authentication (set in .env for production)
API_KEY = os.getenv("RAG_API_KEY")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

# CORS: comma-separated origins, or "*" for development only
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

# Rate limiting
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")  # Requests per minute


# --- App Setup ---

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RAG Sports News API",
    description="Ask questions about sports news with source citations",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - restricted by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Authentication ---

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if authentication is required."""
    if not REQUIRE_AUTH:
        return True

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: REQUIRE_AUTH is true but RAG_API_KEY not set"
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return True


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    question: str = Field(..., min_length=3, max_length=500, description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Max chunks to retrieve")
    min_relevance: float = Field(default=0.3, ge=0, le=1, description="Minimum relevance threshold")


class SourceResponse(BaseModel):
    """A cited source in the response."""
    title: str
    url: str


class QueryResponse(BaseModel):
    """Response from /query endpoint."""
    question: str
    answer: str
    sources: list[SourceResponse]
    context_used: int


class SearchRequest(BaseModel):
    """Request body for /search endpoint."""
    query: str = Field(..., min_length=3, max_length=500, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")


class SearchResultResponse(BaseModel):
    """A single search result."""
    text: str
    title: str
    url: str
    source: str
    relevance_score: float


class SearchResponse(BaseModel):
    """Response from /search endpoint."""
    query: str
    results: list[SearchResultResponse]


# --- Endpoints ---

@app.get("/health")
def health_check():
    """Service health check (no auth required)."""
    return {"status": "healthy"}


@app.get("/stats")
def get_stats(authorized: bool = Depends(verify_api_key)):
    """Get vector database statistics."""
    try:
        stats = get_collection_stats()
        return stats
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")


@app.post("/query", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT)
def query_endpoint(
    request: QueryRequest,
    req: Request,  # Required for rate limiter
    authorized: bool = Depends(verify_api_key),
):
    """
    Answer a question using RAG.

    Retrieves relevant chunks from the vector database,
    then uses GPT-4o-mini to generate an answer with citations.

    Rate limited to prevent abuse.
    """
    try:
        answer: Answer = query(
            question=request.question,
            top_k=request.top_k,
            min_relevance=request.min_relevance,
        )

        return QueryResponse(
            question=answer.question,
            answer=answer.answer,
            sources=[
                SourceResponse(title=s.title, url=s.url)
                for s in answer.sources
            ],
            context_used=answer.context_used,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.post("/search", response_model=SearchResponse)
@limiter.limit(RATE_LIMIT)
def search_endpoint(
    request: SearchRequest,
    req: Request,  # Required for rate limiter
    authorized: bool = Depends(verify_api_key),
):
    """
    Search for relevant chunks without LLM generation.

    Useful for debugging retrieval or building custom UIs.

    Rate limited to prevent abuse.
    """
    try:
        results: list[SearchResult] = search(
            query=request.query,
            top_k=request.top_k,
        )

        return SearchResponse(
            query=request.query,
            results=[
                SearchResultResponse(
                    text=r.text,
                    title=r.title,
                    url=r.url,
                    source=r.source,
                    relevance_score=r.relevance_score,
                )
                for r in results
            ],
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Search failed")


# CLI for testing
if __name__ == "__main__":
    import uvicorn

    print("Starting RAG Sports News API...")
    print(f"Auth required: {REQUIRE_AUTH}")
    print(f"Allowed origins: {ALLOWED_ORIGINS}")
    print(f"Rate limit: {RATE_LIMIT}")
    print("Docs available at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
