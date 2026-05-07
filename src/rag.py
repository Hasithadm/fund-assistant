import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from google.genai.errors import ServerError
import chromadb
from embedder import embed_text

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "funds"

def retrieve(question: str, n_results: int = 2) -> list[str]:
    """Embed the question and find the most similar chunks."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    query_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    chunks = results["documents"][0]
    print(f"\n--- Retrieved {len(chunks)} chunk(s) ---")
    for c in chunks:
        print(f"  > {c[:80]}...")
    return chunks

def answer(question: str) -> str:
    """RAG pipeline: retrieve relevant chunks, then answer from context."""
    chunks = retrieve(question)
    context = "\n\n".join(chunks)

    prompt = f"""You are a private markets fund analyst.
    
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information."

Context: {context}

Question: {question}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    temperature=0.2,  # low temperature = factual, less creative
                ),
                contents=prompt
            )
            break
        except ServerError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))

    return response.text

if __name__ == "__main__":
    questions = [
        "What is the minimum investment for Brookfield Infrastructure Fund V?",
        "Which fund has the highest target return?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(q)}")
        print("-" * 60)