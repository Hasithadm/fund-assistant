import os
from dotenv import load_dotenv
from typing import Annotated
import math

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from embedder import embed_text
import chromadb

load_dotenv()

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "funds"

# ── 1. State ──────────────────────────────────────────────────────────────────
# This is what gets passed between every node in the graph.
# add_messages means new messages are appended, not overwritten — that's memory.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ── 2. Tools ──────────────────────────────────────────────────────────────────
# @tool decorator auto-generates the schema Gemini needs from the docstring

@tool
def search_funds(query: str) -> str:
    """Search for fund information using a natural language query."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    return "\n\n".join(results["documents"][0])

@tool
def get_fund_details(fund_name: str) -> str:
    """Get detailed information about a specific fund by name."""
    return search_funds.invoke({"query": fund_name})

@tool
def calculate_return(principal_usd: float, annual_rate_percent: float, years: int) -> str:
    """Calculate compound investment return given principal, annual rate, and years."""
    final = principal_usd * math.pow(1 + annual_rate_percent / 100, years)
    gain = final - principal_usd
    return (
        f"Principal: ${principal_usd:,.0f}\n"
        f"Rate: {annual_rate_percent}% per year\n"
        f"Years: {years}\n"
        f"Final value: ${final:,.0f}\n"
        f"Total gain: ${gain:,.0f}"
    )

@tool
def list_all_funds() -> str:
    """List all funds available in the database."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    results = collection.get()  # fetch everything, no query needed
    return "\n\n---\n\n".join(results["documents"])

TOOLS = [search_funds, get_fund_details, calculate_return, list_all_funds]

# ── 3. Model ──────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are a private markets fund analyst assistant.
You have tools to search fund information and calculate investment returns.
Always use tools to look up fund data — never guess figures.
If a question needs multiple steps, work through them one at a time.
"""

# ── 4. Nodes ──────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """LLM decides: answer directly OR call a tool."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    print(f"\n  [Agent] {'Calling tool: ' + response.tool_calls[0]['name'] if response.tool_calls else 'Responding directly'}")
    return {"messages": [response]}

def tool_node(state: AgentState) -> AgentState:
    """Execute whatever tool the agent chose."""
    tool_map = {t.name: t for t in TOOLS}
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        tool_fn = tool_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        print(f"  [Tool: {tool_call['name']}] {str(result)[:100]}...")
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": results}

# ── 5. Routing ────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """After agent node: loop to tools, or end?"""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# ── 6. Build the graph ────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")   # always loop back to agent after tool
    return graph.compile()

# ── 7. Multi-turn chat with memory ────────────────────────────────────────────

def chat():
    """Interactive chat — full history passed every turn = memory."""
    app = build_graph()
    history = []

    print("\n=== Fund Research Assistant (LangGraph) ===")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        result = app.invoke({"messages": history})

        # Extract and print the final response
        final_message = result["messages"][-1]
        print(f"\nAssistant: {final_message.content}\n")

        # Update history with the full exchange (for memory)
        history = result["messages"]

if __name__ == "__main__":
    chat()