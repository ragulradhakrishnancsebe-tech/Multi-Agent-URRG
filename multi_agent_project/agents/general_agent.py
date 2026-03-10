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
        history: List[Dict[str, Any]] = None,
    ) -> str:

        try:

            # ✅ Short system prompt (token optimized)
            messages = [
                SystemMessage(
                    content=(
                        "You are a helpful AI assistant. "
                        "Respond clearly using markdown, short sections, and bullet points."
                    )
                )
            ]

            # ✅ Limit history to last 3 turns
            if history:
                history = history[-3:]

                for turn in history:
                    if turn.get("user"):
                        messages.append(
                            HumanMessage(content=turn["user"])
                        )

                    if turn.get("assistant"):
                        messages.append(
                            AIMessage(content=turn["assistant"])
                        )

            # ✅ Current user query
            messages.append(HumanMessage(content=user_query))

            response = self.llm.invoke(messages)

            # ✅ Safe response extraction
            if hasattr(response, "content"):
                return response.content

            return str(response)

        except Exception as e:
            error_msg = f"General agent error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return error_msg