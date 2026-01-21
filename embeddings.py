"""
Embedding and vector storage for RAG.

Key concepts:
- Chunking: Split articles into smaller pieces for precise retrieval
- Embeddings: Convert text to vectors that capture semantic meaning
- Vector DB: Store and search embeddings efficiently
"""

import hashlib
import os
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

from scraper import Article

load_dotenv()


@dataclass
class Chunk:
    """A piece of an article with metadata for citations."""
    text: str
    article_title: str
    article_url: str
    source: str
    chunk_index: int


# --- Chunking ---

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.

    Why chunk?
    - LLMs have context limits
    - Smaller chunks = more precise retrieval
    - Overlap prevents losing context at boundaries

    Args:
        chunk_size: Target characters per chunk (~100 tokens)
        overlap: Characters to repeat between chunks

    This is a simple character-based chunker. Production systems
    might use token-based or semantic chunking.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 chars of chunk
            search_start = max(end - 100, start)
            last_period = text.rfind(". ", search_start, end)
            if last_period > start:
                end = last_period + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def chunk_article(article: Article, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    Convert an article into chunks with metadata.

    Metadata is crucial for RAG - it enables citations!
    Each chunk knows which article it came from.
    """
    text_chunks = chunk_text(article.content, chunk_size, overlap)

    return [
        Chunk(
            text=text,
            article_title=article.title,
            article_url=article.url,
            source=article.source,
            chunk_index=i,
        )
        for i, text in enumerate(text_chunks)
    ]


# --- Embeddings ---

def get_openai_client() -> OpenAI:
    """Get OpenAI client, checking for API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=api_key)


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    text-embedding-3-small:
    - 1536 dimensions
    - Good balance of quality and cost
    - ~$0.02 per 1M tokens

    Batching is important - API calls have overhead, so we
    embed multiple texts in one request.
    """
    client = get_openai_client()

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    return [item.embedding for item in response.data]


# --- Vector Storage (ChromaDB) ---

def get_chroma_collection(collection_name: str = "sports_news"):
    """
    Get or create a ChromaDB collection.

    ChromaDB:
    - Lightweight, runs locally (no server needed)
    - Persists to disk in chroma_db/ folder
    - Handles embedding storage and similarity search

    We use OpenAI's embedding function directly in ChromaDB,
    so it auto-embeds on add() and query().
    """
    # Persist to local directory
    client = chromadb.PersistentClient(path="./chroma_db")

    # Use OpenAI embeddings via ChromaDB's built-in function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef,
        metadata={"description": "Sports news articles for RAG"}
    )

    return collection


def generate_chunk_id(chunk: Chunk) -> str:
    """
    Generate a stable ID for a chunk.

    Using a hash ensures:
    - Same content = same ID (deduplication)
    - No collisions between different chunks
    """
    content = f"{chunk.article_url}:{chunk.chunk_index}:{chunk.text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()


# --- Ingest Pipeline ---

def ingest_articles(articles: list[Article], chunk_size: int = 500) -> int:
    """
    Process articles and store in vector database.

    Pipeline:
    1. Chunk each article
    2. Prepare metadata for citations
    3. Add to ChromaDB (auto-embeds)

    Returns number of chunks added.
    """
    collection = get_chroma_collection()

    all_chunks = []
    for article in articles:
        chunks = chunk_article(article, chunk_size=chunk_size)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks to ingest")
        return 0

    # Prepare data for ChromaDB
    ids = [generate_chunk_id(c) for c in all_chunks]
    documents = [c.text for c in all_chunks]
    metadatas = [
        {
            "title": c.article_title,
            "url": c.article_url,
            "source": c.source,
            "chunk_index": c.chunk_index,
        }
        for c in all_chunks
    ]

    # Upsert to handle duplicates gracefully
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Ingested {len(all_chunks)} chunks from {len(articles)} articles")
    return len(all_chunks)


def get_collection_stats() -> dict:
    """Get info about what's stored in the vector database."""
    collection = get_chroma_collection()
    return {
        "total_chunks": collection.count(),
        "collection_name": collection.name,
    }


# CLI for testing
if __name__ == "__main__":
    from scraper import scrape_feed

    print("=== Embeddings Pipeline Test ===\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env file first")
        print("  cp .env.example .env")
        print("  # Then edit .env with your key")
        exit(1)

    # Scrape some articles (CBS Sports has American sports content)
    print("Step 1: Scraping articles...")
    articles = scrape_feed("cbs_sports", max_articles=5)

    if not articles:
        print("No articles scraped. Check your internet connection.")
        exit(1)

    # Show chunking in action
    print(f"\nStep 2: Chunking {len(articles)} articles...")
    sample_chunks = chunk_article(articles[0])
    print(f"  First article split into {len(sample_chunks)} chunks")
    print(f"  Chunk 1 preview: {sample_chunks[0].text[:100]}...")

    # Ingest into ChromaDB
    print("\nStep 3: Embedding and storing in ChromaDB...")
    num_chunks = ingest_articles(articles)

    # Show stats
    print("\nStep 4: Collection stats")
    stats = get_collection_stats()
    print(f"  Total chunks in database: {stats['total_chunks']}")

    print("\n✓ Pipeline complete! Ready for retrieval.")
