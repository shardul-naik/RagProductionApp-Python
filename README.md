# Production-Ready RAG AI Agent

A production-grade RAG application with **observability, logging, retries, and rate limiting**.

## Tech Stack
- **Orchestration:** Inngest (retryable workflows, monitoring)
- **Frontend:** Streamlit
- **Vector DB:** Qdrant
- **Ingestion:** LlamaIndex
- **AI:** OpenAI

## Key Features
- **Production Workflow:** Automatic retries, execution logging, error handling
- **PDF Ingestion:** Load, chunk, and vectorize documents
- **RAG Inference:** Semantic search with grounded LLM responses
- **Operational Control:** Rate limiting, concurrency control, throttling

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd production-rag-agent
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your OpenAI API key

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Run Inngest dev server
inngest dev

# Launch app
streamlit run app.py
```

## Usage
1. Upload PDFs via Streamlit interface
2. System automatically processes and vectors documents
3. Ask questions in natural language
4. Receive grounded responses with context

## Architecture
```
Streamlit → Inngest → OpenAI API
              ↓
         Qdrant ← LlamaIndex
```

## Production Features
- **Observability:** Full visibility into function runs
- **Reliability:** Automatic retries with exponential backoff
- **Scalability:** Horizontal scaling support
- **Security:** Rate limiting, API key management

---

**License:** MIT
