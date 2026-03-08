from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any


class SummarizeAgent:
    def __init__(self, model_type="gemini"):
        self.llm = get_gemini_llm() if model_type == "gemini" else get_groq_llm()

    def summarize(
        self,
        rag_resp:      str,
        research_resp: str,
        history:       List[Dict[str, Any]] = None,
    ) -> str:
        try:
            # ── Build messages with history ───────────────
            messages = [
                SystemMessage(content=(
                    "You are a summarization assistant.\n\n"

                    "Your task is to combine and summarize information from:\n"
                    "1. RAG responses (information from uploaded documents)\n"
                    "2. Research responses (information from web search)\n\n"

                    "Rules:\n"
                    "- Merge both sources into one coherent answer.\n"
                    "- Remove duplicated information.\n"
                    "- If both sources say similar things, combine them.\n"
                    "- If one source adds extra insight, include it clearly.\n\n"

                    "Formatting Rules:\n"
                    "- Always respond using clean Markdown.\n"
                    "- Use headings (##) when useful.\n"
                    "- Use bullet points for lists.\n"
                    "- Avoid large tables unless necessary.\n"
                    "- Keep the answer concise and readable for chat UI.\n\n"

                    "Conversation Rules:\n"
                    "- You have access to the full conversation history.\n"
                    "- Use history to understand follow-up questions.\n"
                    "- Maintain context from earlier messages.\n\n"

                    "Goal:\n"
                    "Produce a clear, unified answer that reads like a single response."
                ))
            ]

            # ✅ Inject conversation history
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

            # ── Build summary prompt ──────────────────────
            parts = []

            if rag_resp and rag_resp.strip():
                parts.append(f"RAG Response (from uploaded documents):\n{rag_resp.strip()}")

            if research_resp and research_resp.strip():
                parts.append(f"Research Response (from web search):\n{research_resp.strip()}")

            combined = "\n\n".join(parts)

            # ✅ Add current summarization request
            messages.append(
                HumanMessage(content=(
                    f"Please summarize and combine the following responses "
                    f"into a single, clear, well-structured answer:\n\n"
                    f"{combined}"
                ))
            )

            # ── Query LLM ─────────────────────────────────
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            error_msg = f"Summarize agent error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return error_msg