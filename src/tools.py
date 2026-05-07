import math
from embedder import embed_text
import chromadb

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "funds"

def search_funds(query: str) -> str:
    """Search the vector DB for relevant fund information."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    chunks = results["documents"][0]
    return "\n\n".join(chunks)

def get_fund_details(fund_name: str) -> str:
    """Look up a specific fund by name from the vector DB."""
    return search_funds(fund_name)

def calculate_return(
    principal_usd: float,
    annual_rate_percent: float,
    years: int
) -> str:
    """Calculate compound investment return."""
    final = principal_usd * math.pow(1 + annual_rate_percent / 100, years)
    gain = final - principal_usd
    return (
        f"Principal: ${principal_usd:,.0f}\n"
        f"Rate: {annual_rate_percent}% per year\n"
        f"Years: {years}\n"
        f"Final value: ${final:,.0f}\n"
        f"Total gain: ${gain:,.0f}"
    )