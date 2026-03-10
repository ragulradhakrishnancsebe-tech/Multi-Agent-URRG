import os
from tavily import TavilyClient
from env_config import TAVILY_API_KEY
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any


class ResearchAgent:

    def __init__(self, model_type="gemini"):

        self.tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
        self.llm = get_gemini_llm() if model_type == "gemini" else get_groq_llm()

    def query(
        self,
        user_query: str,
        history: List[Dict[str, Any]] = None,
    ) -> str:

        try:

            if not self.tavily:
                return "Search service not configured."

            # 🔎 Advanced Tavily Search
            search_results = self.tavily.search(
                query=user_query,
                max_results=5,
                search_depth="advanced"
            )

            results = search_results.get("results", [])

            if not results:
                return f"No search results found for '{user_query}'."

            combined_text = ""
            reference_links = []

            # 📚 Build context for LLM
            for i, result in enumerate(results, start=1):

                title = result.get("title", "")
                content = (result.get("content") or "")[:500]
                url = result.get("url", "")

                combined_text += (
                    f"[{i}] {title}\n"
                    f"{content}\n\n"
                )

                if url:
                    reference_links.append(f"[{i}] {title}: {url}")

            # 🧠 LLM Instructions
            messages = [
                SystemMessage(
                    content=(
                        "You are a professional research assistant.\n\n"
                        "Use the provided search results to answer the question.\n"
                        "Cite sources using [number] like [1], [2].\n"
                        "Write a structured markdown answer with:\n"
                        "- Short sections\n"
                        "- Bullet points\n"
                        "- Clear explanation\n\n"
                        "Only use the provided search data."
                    )
                )
            ]

            # 📜 Add limited conversation history
            if history:

                history = history[-3:]

                for turn in history:

                    if turn.get("user"):
                        messages.append(HumanMessage(content=turn["user"]))

                    if turn.get("assistant"):
                        messages.append(AIMessage(content=turn["assistant"]))

            # ❓ Add research question
            messages.append(
                HumanMessage(
                    content=(
                        f"Search Results:\n{combined_text}\n"
                        f"Question: {user_query}"
                    )
                )
            )

            # 🤖 LLM Response
            response = self.llm.invoke(messages)

            summary = response.content if hasattr(response, "content") else str(response)

            final_response = summary

            # 🔗 Add references
            if reference_links:
                final_response += "\n\n---\n### References\n"
                for link in reference_links:
                    final_response += f"{link}\n"

            return final_response

        except Exception as e:

            error_msg = f"Research agent error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return error_msg