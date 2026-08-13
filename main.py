import os
import uuid
import logging
import asyncio
from typing import Any, cast
from fastapi import FastAPI
from dotenv import load_dotenv
import inngest
import inngest.fast_api
from openai import OpenAI

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

# Initialize Vector DB globally
qdrant_store = QdrantStorage()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False
)

# ------------------------------------------------------------------
# 1. Ingestion Workflow
# ------------------------------------------------------------------
@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context) -> dict[str, Any]:
    # Fix 1: Validate pdf_path is strictly a string for Pylance
    raw_pdf_path = ctx.event.data.get("pdf_path")
    if not raw_pdf_path or not isinstance(raw_pdf_path, str):
        return {"error": "Missing or invalid 'pdf_path' in event data."}
    
    pdf_path: str = raw_pdf_path
    
    raw_source_id = ctx.event.data.get("source_id")
    source_id: str = raw_source_id if isinstance(raw_source_id, str) and raw_source_id else pdf_path

    # Step 1: Load and Chunk PDF
    async def _load() -> list[str]:
        return await asyncio.to_thread(load_and_chunk_pdf, pdf_path)

    chunks: list[str] = await ctx.step.run("load-and-chunk", _load)

    if not chunks:
        return {"ingested": 0, "message": "No text extracted from PDF."}

    # Step 2: Embed and Upsert
    async def _upsert() -> dict[str, int]:
        vecs = await asyncio.to_thread(embed_texts, chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        qdrant_store.upsert(ids, vecs, payloads)
        return {"ingested": len(chunks)}

    ingested: dict[str, int] = await ctx.step.run("embed-and-upsert", _upsert)
    return ingested


# ------------------------------------------------------------------
# 2. Query Workflow
# ------------------------------------------------------------------
@inngest_client.create_function(
    fn_id="rag-query-pdf-ai",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> dict[str, Any]:
    # Fix 2: Validate question is strictly a string
    raw_question = ctx.event.data.get("question")
    if not raw_question or not isinstance(raw_question, str):
        return {"error": "Missing or invalid 'question' in event data."}
    
    question: str = raw_question
    
    # Fix 3: Validate top_k is strictly an integer
    raw_top_k = ctx.event.data.get("top_k", 5)
    top_k: int = int(raw_top_k) if isinstance(raw_top_k, (int, str)) else 5

    # Step 1: Search Vector DB
    async def _search() -> dict[str, Any]:
        query_vecs = await asyncio.to_thread(embed_texts, [question])
        query_vec = query_vecs[0]
        
        found = qdrant_store.search(query_vec, top_k)
        
        contexts = cast(list[str], found.get("contexts", []))
        raw_sources = cast(list[Any], found.get("sources", []))
        valid_sources: list[str] = [str(s) for s in raw_sources if s]
        
        return {"contexts": contexts, "sources": valid_sources}

    found: dict[str, Any] = await ctx.step.run("embed-and-search", _search)

    contexts: list[str] = cast(list[str], found.get("contexts", []))
    sources: list[str] = cast(list[str], found.get("sources", []))

    # Step 2: LLM Inference
    async def _llm() -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        context_block = "\n\n".join(f"- {c}" for c in contexts)
        user_content = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context above."
        )

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        )
        
        # Fix 4: Safely narrow None -> str before calling .strip()
        content = response.choices[0].message.content
        return content.strip() if content is not None else ""

    answer: str = await ctx.step.run("llm-answer", _llm)
    
    return {
        "answer": answer,
        "sources": sources,
        "num_contexts": len(contexts)
    }


# ------------------------------------------------------------------
# FastAPI Server Initialization
# ------------------------------------------------------------------
app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai]
)