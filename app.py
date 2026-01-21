"""
Streamlit UI for RAG Sports News.

A simple chat-like interface for asking questions about sports news.
Displays answers with source citations.

Security features:
- Session-based rate limiting
- Sanitized error messages
"""

import time
import streamlit as st

from query import query
from embeddings import get_collection_stats

# --- Rate Limiting ---

RATE_LIMIT_QUERIES = 10  # Max queries per window
RATE_LIMIT_WINDOW = 60  # Window in seconds (1 minute)


def check_rate_limit() -> bool:
    """
    Check if user has exceeded rate limit.
    Returns True if allowed, False if rate limited.
    """
    now = time.time()

    # Initialize session state
    if "query_timestamps" not in st.session_state:
        st.session_state.query_timestamps = []

    # Remove timestamps outside the window
    st.session_state.query_timestamps = [
        ts for ts in st.session_state.query_timestamps
        if now - ts < RATE_LIMIT_WINDOW
    ]

    # Check if under limit
    if len(st.session_state.query_timestamps) >= RATE_LIMIT_QUERIES:
        return False

    # Record this query
    st.session_state.query_timestamps.append(now)
    return True


def get_rate_limit_status() -> str:
    """Get human-readable rate limit status."""
    if "query_timestamps" not in st.session_state:
        return f"{RATE_LIMIT_QUERIES} queries remaining"

    remaining = RATE_LIMIT_QUERIES - len(st.session_state.query_timestamps)
    return f"{remaining} queries remaining this minute"


# --- Page Config ---

st.set_page_config(
    page_title="RAG Sports News",
    page_icon="🏈",
    layout="centered",
)

st.title("🏈 RAG Sports News")
st.caption("Ask questions about recent sports news. Answers include source citations.")

# Sidebar with stats and settings
with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More sources = more context, but slower and potentially noisier",
    )

    min_relevance = st.slider(
        "Minimum relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher = stricter filtering, may miss relevant results",
    )

    st.divider()
    st.header("Database Stats")

    try:
        stats = get_collection_stats()
        st.metric("Total Chunks", stats["total_chunks"])
        st.caption(f"Collection: {stats['collection_name']}")
    except Exception:
        st.warning("Database loading...")

    st.divider()
    st.caption(get_rate_limit_status())

# Main chat interface
question = st.text_input(
    "Ask a question about sports news:",
    placeholder="Who was elected to the Baseball Hall of Fame?",
)

if st.button("Ask", type="primary") or (question and st.session_state.get("last_question") != question):
    if question:
        # Check rate limit before processing
        if not check_rate_limit():
            st.error("Rate limit exceeded. Please wait a minute before asking more questions.")
        else:
            st.session_state.last_question = question

            with st.spinner("Searching and generating answer..."):
                try:
                    answer = query(
                        question=question,
                        top_k=top_k,
                        min_relevance=min_relevance,
                    )

                    # Display answer
                    st.subheader("Answer")
                    st.write(answer.answer)

                    # Display sources
                    if answer.sources:
                        st.subheader(f"Sources ({answer.context_used} used)")
                        for i, source in enumerate(answer.sources, 1):
                            with st.expander(f"{i}. {source.title[:80]}..."):
                                st.markdown(f"[Read full article]({source.url})")
                    else:
                        st.info("No sources met the relevance threshold.")

                except Exception:
                    st.error("Unable to process your question. Please try again.")
    else:
        st.warning("Please enter a question.")

# Example questions
st.divider()
st.subheader("Try these example questions:")
examples = [
    "Who was elected to the Baseball Hall of Fame this year?",
    "What's happening with the NCAA lawsuit?",
    "Where can I watch UFC fights?",
]

cols = st.columns(len(examples))
for i, example in enumerate(examples):
    if cols[i].button(example, key=f"example_{i}"):
        st.session_state.last_question = None  # Reset to allow new query
        st.rerun()
