import os
from pathlib import Path
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def get_openai_client():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def load_and_chunk_pdf(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found at path: {path}")
        
    # Updated: Pass file=Path(path) or file_path=file_path directly
    docs = PDFReader().load_data(file=file_path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    
    if not texts:
        return []

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
        
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

