"""
RAG query pipeline - combines retrieval with LLM generation.

Key concepts:
- Context injection: Put retrieved chunks into the prompt
- Prompt engineering: Instruct LLM to use context and cite sources
- Grounded generation: Answer based on retrieved facts, not hallucination
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from retriever import search, SearchResult

load_dotenv()


@dataclass
class Source:
    """A cited source for the answer."""
    title: str
    url: str


@dataclass
class Answer:
    """RAG response with answer and citations."""
    question: str
    answer: str
    sources: list[Source]
    context_used: int  # Number of chunks used


# System prompt instructs the LLM how to behave
SYSTEM_PROMPT = """You are a sports news assistant. Answer questions based ONLY on the provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have information about that in my current news sources."
3. Be concise and direct
4. When citing facts, mention the source naturally (e.g., "According to CBS Sports...")
5. Don't make up information not in the context"""


def build_context(results: list[SearchResult]) -> str:
    """
    Format retrieved chunks into context for the prompt.

    Each chunk is labeled with its source for citation.
    """
    if not results:
        return "No relevant context found."

    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[Source {i}: {r.title}]\n{r.text}\n")

    return "\n---\n".join(context_parts)


def query(question: str, top_k: int = 5, min_relevance: float = 0.3) -> Answer:
    """
    Answer a question using RAG.

    Pipeline:
    1. Retrieve relevant chunks from vector database
    2. Filter by relevance threshold
    3. Build context from chunks
    4. Call LLM with context + question
    5. Return answer with source citations

    Args:
        question: User's natural language question
        top_k: Max chunks to retrieve
        min_relevance: Minimum relevance score (0-1)

    Returns:
        Answer object with response and cited sources
    """
    # Step 1: Retrieve relevant chunks
    results = search(question, top_k=top_k)

    # Step 2: Filter by relevance
    relevant_results = [r for r in results if r.relevance_score >= min_relevance]

    # Step 3: Build context
    context = build_context(relevant_results)

    # Step 4: Build the prompt
    user_prompt = f"""Context:
{context}

Question: {question}

Answer the question based on the context above. Cite your sources."""

    # Step 5: Call LLM
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # Lower = more factual, less creative
        max_tokens=500,
    )

    answer_text = response.choices[0].message.content

    # Step 6: Build response with sources
    sources = [
        Source(title=r.title, url=r.url)
        for r in relevant_results
    ]

    return Answer(
        question=question,
        answer=answer_text,
        sources=sources,
        context_used=len(relevant_results),
    )


def format_answer(answer: Answer) -> str:
    """Format an Answer for display."""
    lines = [
        f"Q: {answer.question}",
        "",
        f"A: {answer.answer}",
        "",
        f"Sources ({answer.context_used} used):",
    ]

    for i, source in enumerate(answer.sources, 1):
        lines.append(f"  {i}. {source.title}")
        lines.append(f"     {source.url}")

    return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    print("=== RAG Query Test ===\n")

    test_questions = [
        "Who was elected to the Baseball Hall of Fame this year?",
        "What's the story with the NCAA lawsuit?",
        "Where can I watch UFC fights?",
        "How did the Lakers do last night?",  # Not in our data - tests "no info" response
    ]

    for question in test_questions:
        print("=" * 60)
        answer = query(question)
        print(format_answer(answer))
        print()
