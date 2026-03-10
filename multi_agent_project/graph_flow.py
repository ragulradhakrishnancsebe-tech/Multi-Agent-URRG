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
    query: str
    history: List[Dict[str, Any]]
    uploaded_docs: List[str]

    category: str
    rag_response: str
    research_response: str
    general_response: str

    summary: str


# ── Global memory ─────────────────────────────────────────
checkpointer = MemorySaver()


# ── Helper: limit memory size ─────────────────────────────
def trim_history(history, limit=4):
    if not history:
        return []
    return history[-limit:]


# ── Build graph ───────────────────────────────────────────
def build_graph(model_type="gemini"):

    understanding_agent = QueryUnderstandingAgent(model_type)
    rag_agent = RAGAgent(model_type)
    research_agent = ResearchAgent(model_type)
    general_agent = GeneralAgent(model_type)
    summarize_agent = SummarizeAgent(model_type)

    graph = StateGraph(AgentState)

    # ── Understanding Node ────────────────────────────────
    def understanding_node(state):

        try:
            history = trim_history(state.get("history", []))

            category = understanding_agent.analyze_query(
                state["query"],
                history
            )

            return {"category": category}

        except Exception as e:
            print(f"Understanding agent error: {e}")
            return {"category": "general"}


    # ── RAG Node ──────────────────────────────────────────
    def rag_node(state):

        if state.get("category") != "rag":
            return {"rag_response": ""}

        try:

            history = trim_history(state.get("history", []))

            rag_response = rag_agent.query(
                state["query"],
                state.get("uploaded_docs", []),
                history
            )

            return {"rag_response": rag_response}

        except Exception as e:
            print(f"RAG agent error: {e}")
            return {"rag_response": ""}


    # ── Research Node ─────────────────────────────────────
    def research_node(state):

        if state.get("category") != "research":
            return {"research_response": ""}

        try:

            history = trim_history(state.get("history", []))

            research_response = research_agent.query(
                state["query"],
                history
            )

            return {"research_response": research_response}

        except Exception as e:
            print(f"Research agent error: {e}")
            return {"research_response": ""}


    # ── General Node ──────────────────────────────────────
    def general_node(state):

        if state.get("category") != "general":
            return {"general_response": ""}

        try:

            history = trim_history(state.get("history", []))

            general_response = general_agent.respond(
                state["query"],
                history
            )

            return {"general_response": general_response}

        except Exception as e:
            print(f"General agent error: {e}")
            return {"general_response": ""}


    # ── Summary Node (Conversation Memory) ────────────────
    def summarize_node(state):

        try:

            rag_resp = state.get("rag_response", "")
            research_resp = state.get("research_response", "")
            general_resp = state.get("general_response", "")

            history = state.get("history", [])

            # Final response
            response = rag_resp or research_resp or general_resp

            if not response:
                return {"summary": "", "history": history}

            # Update history
            history.append({
                "query": state["query"],
                "response": response
            })

            # Summarize when history grows
            if len(history) > 6:

                summary = summarize_agent.summarize(
                    rag_resp,
                    research_resp,
                    trim_history(history)
                )

                history = history[-2:]  # keep latest messages only

                return {
                    "summary": summary,
                    "history": history
                }

            return {"history": history}

        except Exception as e:
            print(f"Summarize agent error: {e}")
            return {"summary": "", "history": state.get("history", [])}


    # ── Register nodes ────────────────────────────────────
    graph.add_node("understanding_node", understanding_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("research_node", research_node)
    graph.add_node("general_node", general_node)
    graph.add_node("summarize_node", summarize_node)


    # ── Edges ─────────────────────────────────────────────
    graph.set_entry_point("understanding_node")

    graph.add_edge("understanding_node", "rag_node")
    graph.add_edge("understanding_node", "research_node")
    graph.add_edge("understanding_node", "general_node")

    graph.add_edge("rag_node", "summarize_node")
    graph.add_edge("research_node", "summarize_node")
    graph.add_edge("general_node", "summarize_node")

    graph.add_edge("summarize_node", END)


    # ── Compile graph with memory ─────────────────────────
    return graph.compile(checkpointer=checkpointer)