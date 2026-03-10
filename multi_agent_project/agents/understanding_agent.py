from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any


class QueryUnderstandingAgent:

    def __init__(self, model_type: str = "gemini"):
        """
        Initialize router agent with selected LLM.
        """

        if model_type == "gemini":
            self.llm = get_gemini_llm()
        else:
            self.llm = get_groq_llm()

    def analyze_query(
        self,
        query: str,
        history: List[Dict[str, Any]] = None,
    ) -> str:
        """
        Classify user query into:
        rag | research | general
        """

        try:

            # ── Short system prompt (token optimized) ──
            messages = [
                SystemMessage(
                    content=(
                        "Classify the user's query into one category:\n"
                        "rag -> document/PDF/file questions\n"
                        "research -> latest news or real-time info\n"
                        "general -> normal conversation or knowledge\n\n"
                        "Return ONLY one word: rag, research, or general."
                    )
                )
            ]

            # ── Add limited history (last 3 turns only) ──
            if history:

                history = history[-3:]

                for turn in history:

                    if not isinstance(turn, dict):
                        continue

                    if turn.get("user"):
                        messages.append(
                            HumanMessage(content=str(turn["user"]))
                        )

                    if turn.get("assistant"):
                        messages.append(
                            AIMessage(content=str(turn["assistant"]))
                        )

            # ── Add current user query ──
            messages.append(HumanMessage(content=query))

            # ── Call LLM ──
            response = self.llm.invoke(messages)

            if not response:
                return "general"

            content = getattr(response, "content", "")
            if not content:
                return "general"

            category = content.lower().strip()

            # ── Safe category extraction ──
            if "rag" in category:
                return "rag"

            if "research" in category:
                return "research"

            if "general" in category:
                return "general"

            # fallback
            return "general"

        except Exception as e:

            print(f"QueryUnderstandingAgent error: {type(e).__name__}: {e}")

            # safe fallback
            return "general"