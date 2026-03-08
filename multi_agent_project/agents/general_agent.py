from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any


class GeneralAgent:
    def __init__(self, model_type="gemini"):
        self.llm = get_gemini_llm() if model_type == "gemini" else get_groq_llm()

    def respond(
        self,
        user_query: str,
        history:    List[Dict[str, Any]] = None,
    ) -> str:
        try:
            # ── System prompt ─────────────────────────────
            messages = [
                SystemMessage(content=(
                    "You are a helpful, friendly AI assistant.\n\n"

                    "Formatting Rules:\n"
                    "- Always respond in clean Markdown.\n"
                    "- Use headings (##) for main sections.\n"
                    "- Use bullet points instead of long paragraphs.\n"
                    "- Avoid large tables unless the user explicitly asks for a table.\n"
                    "- Keep responses clear and readable for chat UI.\n"
                    "- Use short paragraphs.\n\n"

                    "Conversation Rules:\n"
                    "- You have access to the full conversation history.\n"
                    "- Use history to answer follow-up questions correctly.\n"
                    "- Remember details the user mentioned earlier.\n"
                    "- Maintain natural and coherent conversation.\n"
                    "- Never say you don't know something already mentioned in history."
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

            # ✅ Add current query
            messages.append(HumanMessage(content=user_query))

            # ── Invoke LLM ────────────────────────────────
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            error_msg = f"General agent error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return error_msg