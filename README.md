# Fund Research Assistant

An agentic AI assistant for private markets fund research.

Built as a learning project covering:
- RAG pipelines with ChromaDB vector database
- Semantic search with Gemini embeddings
- Tool use and function calling
- LangGraph agent loops with multi-turn memory
- Streamlit UI

## Stack
- **LLM** — Google Gemini 2.5 Flash
- **Orchestration** — LangGraph
- **Vector DB** — ChromaDB
- **Embeddings** — Gemini Embedding
- **UI** — Streamlit

## Architecture

```
User question
     ↓
LangGraph agent loop
     ↓
Gemini decides: search_funds / get_fund_details / calculate_return / list_all_funds
     ↓
Tool executes → result passed back to agent
     ↓
Agent loops until ready → final answer
     ↓
Streamlit UI displays response
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
python src/embedder.py # build the vector store
streamlit run src/app.py
```

## Project structure

```
src/
  app.py           # Streamlit UI
  graph_agent.py   # LangGraph agent loop + memory
  embedder.py      # Chunking + ChromaDB ingestion
  rag.py           # Basic RAG pipeline
  tools.py         # Tool functions
  agent.py         # Phase 3 single-turn agent
  models.py        # Pydantic output schemas
  prompts.py       # System prompts
data/
  funds.txt        # Fund documents
```
