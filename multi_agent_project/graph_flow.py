from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any
from agents.understanding_agent import QueryUnderstandingAgent
from agents.rag_agent import RAGAgent
from agents.research_agent import ResearchAgent
from agents.general_agent import GeneralAgent
from agents.summarize_agent import SummarizeAgent


# ── State ────────────────────────────────────────────────
class AgentState(TypedDict):
    query:             str
    history:           List[Dict[str, Any]]
    uploaded_docs:     List[str]
    category:          str
    rag_response:      str
    research_response: str
    general_response:  str
    summary:           str


# ── Global checkpointer (one instance, shared across sessions) ──
checkpointer = MemorySaver()


# ── Build graph ──────────────────────────────────────────
def build_graph(model_type="gemini"):

    understanding_agent = QueryUnderstandingAgent(model_type)
    rag_agent           = RAGAgent(model_type)
    research_agent      = ResearchAgent(model_type)
    general_agent       = GeneralAgent(model_type)
    summarize_agent     = SummarizeAgent(model_type)

    graph = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────

    def understanding_node(state):
        try:
            category = understanding_agent.analyze_query(
                state["query"],
                state.get("history", [])        # ✅ history passed
            )
            return {"category": category}
        except Exception as e:
            print(f"Understanding agent error: {e}")
            return {"category": "general"}

    def rag_node(state):
        try:
            if state.get("category") == "rag":
                rag_response = rag_agent.query(
                    state["query"],
                    state.get("uploaded_docs", []),
                    state.get("history", [])    # ✅ history passed
                )
                return {"rag_response": rag_response}
        except Exception as e:
            print(f"RAG agent error: {e}")
        return {"rag_response": ""}

    def research_node(state):
        try:
            if state.get("category") == "research":
                research_response = research_agent.query(
                    state["query"],
                    state.get("history", [])    # ✅ history passed
                )
                return {"research_response": research_response}
        except Exception as e:
            print(f"Research agent error: {e}")
        return {"research_response": ""}

    def general_node(state):
        try:
            if state.get("category") == "general":
                general_response = general_agent.respond(
                    state["query"],
                    state.get("history", [])    # ✅ history passed
                )
                return {"general_response": general_response}
        except Exception as e:
            print(f"General agent error: {e}")
        return {"general_response": ""}

    def summarize_node(state):
        try:
            rag_resp      = state.get("rag_response",      "")
            research_resp = state.get("research_response", "")

            if rag_resp or research_resp:
                summary = summarize_agent.summarize(
                    rag_resp,
                    research_resp,
                    state.get("history", [])    # ✅ history passed (was missing)
                )
                return {"summary": summary}

        except Exception as e:
            print(f"Summarize agent error: {e}")

        return {"summary": ""}

    # ── Register nodes ───────────────────────────────────
    graph.add_node("understanding_node", understanding_node)
    graph.add_node("rag_node",           rag_node)
    graph.add_node("research_node",      research_node)
    graph.add_node("general_node",       general_node)
    graph.add_node("summarize_node",     summarize_node)

    # ── Edges ────────────────────────────────────────────
    graph.set_entry_point("understanding_node")

    graph.add_edge("understanding_node", "rag_node")
    graph.add_edge("understanding_node", "research_node")
    graph.add_edge("understanding_node", "general_node")

    graph.add_edge("rag_node",      "summarize_node")
    graph.add_edge("research_node", "summarize_node")

    graph.add_edge("general_node",   END)
    graph.add_edge("summarize_node", END)

    # ── Compile WITH checkpointer ────────────────────────
    return graph.compile(checkpointer=checkpointer)