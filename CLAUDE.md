# RAG Sports News

A Retrieval-Augmented Generation system for sports news Q&A with source citations.

## Project Type
Standalone AI learning project — "Enterprise RAG Decision System" (portfolio must-have)

## Tech Stack
- **LLM:** GPT-4o-mini (OpenAI)
- **Embeddings:** text-embedding-3-small (OpenAI)
- **Vector DB:** ChromaDB (local)
- **UI:** Streamlit
- **API:** FastAPI
- **News Sources:** RSS feeds (ESPN, Yahoo Sports)

## Architecture

```
News Sources → Scraper → Chunker → Embeddings → ChromaDB
                                                    ↓
User Question → Embed Query → Retrieve Top-K → LLM + Context → Answer + Citations
```

## Project Structure

```
rag-sports-news/
├── CLAUDE.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py              # Streamlit UI
├── main.py             # FastAPI service
├── scraper.py          # News ingestion
├── embeddings.py       # Embedding + ChromaDB storage
├── retriever.py        # Vector search
├── query.py            # RAG logic
└── examples/           # Sample Q&A for portfolio
```

## Key Skills
- Text embeddings and vector similarity search
- Document chunking strategies
- Retrieval tuning (top-k, thresholds)
- Context injection and prompt construction
- Production patterns: caching, batch embedding

## Deployment
- **Demo:** Streamlit Cloud (free)
- **Production:** Optional AWS deployment later

## Environment Variables
```
OPENAI_API_KEY=sk-...
```

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit UI
streamlit run app.py

# Run FastAPI
uvicorn main:app --reload
```

## Portfolio Checklist
- [ ] README with architecture diagram
- [ ] Working Streamlit demo
- [ ] Example Q&A with citations in /examples
- [ ] Deployed to Streamlit Cloud
- [ ] Demo video
