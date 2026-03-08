from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any


class QueryUnderstandingAgent:
    def __init__(self, model_type="gemini"):
        self.llm = get_gemini_llm() if model_type == "gemini" else get_groq_llm()

    def analyze_query(
        self,
        query:   str,
        history: List[Dict[str, Any]] = None,
    ) -> str:
        try:
            # ── System prompt ─────────────────────────────
            messages = [
                SystemMessage(content=(
                    "You are a query classification assistant.\n\n"

                    "Your task is to classify the user's latest query into exactly one category:\n\n"

                    "- rag      : questions about uploaded documents, PDFs, files, or document content\n"
                    "- research : questions that require internet search, recent news, current events, "
                    "or real-time information\n"
                    "- general  : greetings, casual conversation, general knowledge, explanations, "
                    "or anything not related to documents or real-time web search\n\n"

                    "Important rules:\n"
                    "- Use conversation history to understand follow-up questions.\n"
                    "- If the user refers to 'the document', 'this PDF', or 'the file', choose rag.\n"
                    "- If the user asks about 'latest', 'today', 'recent', or news, choose research.\n"
                    "- Otherwise choose general.\n\n"

                    "Output rules:\n"
                    "- Respond with ONLY one word: rag, research, or general.\n"
                    "- Do NOT explain your answer.\n"
                    "- Do NOT include punctuation or extra text."
                ))
            ]

            # ✅ Inject conversation history for context-aware classification
            if history:
                for turn in history:
                    if isinstance(turn, dict):
                        if turn.get("user"):
                            messages.append(
                                HumanMessage(content=turn["user"])
                            )
                        if turn.get("assistant"):
                            messages.append(
                                AIMessage(content=turn["assistant"])
                            )

            # ✅ Add current query to classify
            messages.append(HumanMessage(content=query))

            # ── Invoke LLM ────────────────────────────────
            response = self.llm.invoke(messages)
            category = response.content.lower().strip()

            # ── Extract category word safely ──────────────
            if "rag" in category:
                return "rag"
            elif "research" in category:
                return "research"
            else:
                return "general"

        except Exception as e:
            print(f"QueryUnderstandingAgent error: {e}")
            return "general"   # safe fallback