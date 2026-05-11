import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph_agent import build_graph
from agent import ask
from google.genai.errors import ClientError

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fund Research Assistant", page_icon="📈", layout="centered"
)

st.title("📈 Fund Research Assistant")

# ── Mode selector ─────────────────────────────────────────────────────────────

mode = st.radio(
    "Agent mode",
    ["Gemini SDK — genai.Tool", "LangGraph — @tool decorator"],
    horizontal=True,
)

if "active_mode" not in st.session_state:
    st.session_state.active_mode = mode

if st.session_state.active_mode != mode:
    st.session_state.active_mode = mode
    st.session_state.history = []
    st.session_state.pop("graph", None)

if mode == "Gemini SDK — genai.Tool":
    st.caption("Raw Gemini function calling — stateless, no conversation memory")
else:
    st.caption("LangGraph agent loop — stateful, remembers the full conversation")

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if mode == "LangGraph — @tool decorator" and "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛠️ Tools available")
    st.markdown("""
- `search_funds` — semantic search over fund docs
- `get_fund_details` — look up a fund by name
- `calculate_return` — compound return calculator
- `list_all_funds` — all funds in DB *(LangGraph only)*
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
    if mode == "LangGraph — @tool decorator":
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            with st.chat_message("assistant"):
                content = msg.content
                if isinstance(content, list):
                    text = " ".join(
                        b["text"]
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    text = content
                st.write(text)
    else:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# ── Input handling ────────────────────────────────────────────────────────────

user_input = None
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

typed = st.chat_input("Ask about funds, returns, strategies...")
if typed:
    user_input = typed

# ── Agent call ────────────────────────────────────────────────────────────────

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    if mode == "LangGraph — @tool decorator":
        st.session_state.history.append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.graph.invoke(
                    {"messages": st.session_state.history}
                )
                final_message = result["messages"][-1]
                content = final_message.content
                if isinstance(content, list):
                    response_text = " ".join(
                        b["text"]
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    response_text = content
            st.write(response_text)
        st.session_state.history = result["messages"]

    else:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_text = ask(user_input)
                except ClientError as e:
                    if e.code == 429:
                        response_text = (
                            "Rate limit reached. Please wait a moment and try again."
                        )
                    else:
                        raise
            st.write(response_text)
        st.session_state.history.append({"role": "assistant", "content": response_text})
