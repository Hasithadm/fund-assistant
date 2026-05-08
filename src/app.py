import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph_agent import build_graph

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fund Research Assistant",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Fund Research Assistant")
st.caption("Agentic AI — powered by Gemini + LangGraph + ChromaDB")

# ── Session state (memory across Streamlit reruns) ────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛠️ Tools available")
    st.markdown("""
- `search_funds` — semantic search over fund docs
- `get_fund_details` — look up a fund by name
- `calculate_return` — compound return calculator
- `list_all_funds` — show all funds in the DB
""")
    st.markdown("---")
    st.markdown("### 💡 Try asking")
    suggestions = [
        "What funds do you know about?",
        "What is the minimum investment for Sequoia?",
        "If I invest $1M in Brookfield at their target return for 10 years, what do I get?",
        "Which fund has the lowest risk?",
        "Compare the target returns of all three funds",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True):
            st.session_state.pending_input = s

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)

# ── Input handling ────────────────────────────────────────────────────────────

# Handle sidebar suggestion buttons
user_input = None
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# Handle typed input
typed = st.chat_input("Ask about funds, returns, strategies...")
if typed:
    user_input = typed

# ── Agent call ────────────────────────────────────────────────────────────────

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.history.append(HumanMessage(content=user_input))

    # Run agent with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.graph.invoke(
                {"messages": st.session_state.history}
            )
            final_message = result["messages"][-1]
            content = final_message.content
            if isinstance(content, list):
                response_text = " ".join(
                    block["text"] for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                response_text = content

        st.write(response_text)

    # Update history for memory
    st.session_state.history = result["messages"]