import os
import hashlib
from google import genai
from dotenv import load_dotenv
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "funds"

def chunk_text(text: str) -> list[str]:
    """Split by double newline — each fund becomes its own chunk."""
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks

def embed_text(text: str) -> list[float]:
    """Get embedding vector from Gemini."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def build_vector_store(file_path: str):
    """Chunk, embed, and store all fund data into ChromaDB."""
    with open(file_path, "r") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Found {len(chunks)} chunks")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        embedding = embed_text(chunk)
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"index": i}]
        )
        print(f"  Upserted chunk {i}: {chunk[:60]}...")

    print(f"\nVector store built at {CHROMA_PATH}")
    return collection

if __name__ == "__main__":
    build_vector_store("data/funds.txt")