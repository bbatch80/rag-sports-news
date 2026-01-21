"""
Vector similarity search for RAG retrieval.

Key concepts:
- Query embedding: Convert question to same vector space as documents
- Similarity search: Find chunks closest to query in vector space
- Top-k: Return the k most relevant results
- Distance: Lower = more similar (cosine distance)
"""

from dataclasses import dataclass

from embeddings import get_chroma_collection


@dataclass
class SearchResult:
    """A retrieved chunk with metadata for citations."""
    text: str
    title: str
    url: str
    source: str
    distance: float  # Lower = more similar

    @property
    def relevance_score(self) -> float:
        """Convert distance to 0-1 relevance score (higher = better)."""
        # ChromaDB uses L2 distance by default; this gives intuitive scores
        return max(0, 1 - self.distance)


def search(query: str, top_k: int = 5) -> list[SearchResult]:
    """
    Search for chunks relevant to a query.

    How it works:
    1. ChromaDB embeds your query using the same model as documents
    2. Finds the top_k closest document vectors (cosine similarity)
    3. Returns chunks with metadata for citations

    Args:
        query: Natural language question or search terms
        top_k: Number of results to return (default 5)

    Returns:
        List of SearchResult objects, sorted by relevance
    """
    collection = get_chroma_collection()

    # ChromaDB handles query embedding automatically
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Parse results into SearchResult objects
    search_results = []

    # Results come back as lists of lists (one per query)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        search_results.append(SearchResult(
            text=doc,
            title=meta.get("title", "Unknown"),
            url=meta.get("url", ""),
            source=meta.get("source", "Unknown"),
            distance=dist,
        ))

    return search_results


def search_with_threshold(
    query: str,
    top_k: int = 5,
    min_relevance: float = 0.3
) -> list[SearchResult]:
    """
    Search with a minimum relevance threshold.

    Useful for filtering out low-quality matches. If no results
    meet the threshold, returns empty list rather than bad matches.
    """
    results = search(query, top_k=top_k)
    return [r for r in results if r.relevance_score >= min_relevance]


def format_results(results: list[SearchResult]) -> str:
    """Format search results for display."""
    if not results:
        return "No relevant results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"\n{'='*60}")
        lines.append(f"Result {i} (relevance: {r.relevance_score:.2f})")
        lines.append(f"Source: {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append(f"{'='*60}")
        lines.append(r.text)

    return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    print("=== Retriever Test ===\n")

    # Test queries
    test_queries = [
        "Who was elected to the Baseball Hall of Fame?",
        "What happened with the NCAA lawsuit?",
        "How can I watch UFC fights?",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)

        results = search(query, top_k=3)

        if results:
            for i, r in enumerate(results, 1):
                print(f"\n{i}. [{r.relevance_score:.2f}] {r.title[:50]}...")
                print(f"   {r.text[:150]}...")
        else:
            print("No results found.")

        print()
