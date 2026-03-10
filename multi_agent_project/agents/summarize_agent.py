from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any


class SummarizeAgent:

    def __init__(self, model_type="gemini"):

        self.llm = get_gemini_llm() if model_type == "gemini" else get_groq_llm()

    def summarize(
        self,
        rag_resp: str,
        research_resp: str,
        history: List[Dict[str, Any]] = None,
    ) -> str:

        try:

            # ── Skip summarizer if only one source ──

            if rag_resp and not research_resp:
                return rag_resp

            if research_resp and not rag_resp:
                return research_resp

            if not rag_resp and not research_resp:
                return ""

            # ── Short system prompt ──

            messages = [
                SystemMessage(
                    content=(
                        "Combine document and web information into one clear answer. "
                        "Remove duplicate information. "
                        "Respond clearly using markdown, short sections, and bullet points."
                    )
                )
            ]

            # ── Limit history ──

            if history:

                history = history[-2:]

                for turn in history:

                    if turn.get("user"):
                        messages.append(
                            HumanMessage(content=turn["user"])
                        )

                    if turn.get("assistant"):
                        messages.append(
                            AIMessage(content=turn["assistant"])
                        )

            # ── Compress responses ──

            rag_text = (rag_resp or "")[:800]
            research_text = (research_resp or "")[:800]

            messages.append(
                HumanMessage(
                    content=(
                        f"Document info:\n{rag_text}\n\n"
                        f"Web info:\n{research_text}\n\n"
                        f"Create a single clear answer."
                    )
                )
            )

            response = self.llm.invoke(messages)

            if hasattr(response, "content"):
                return response.content

            return str(response)

        except Exception as e:

            error_msg = f"Summarize agent error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return error_msg