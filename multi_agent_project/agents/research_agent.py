import os
from tavily import TavilyClient
from env_config import TAVILY_API_KEY
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any


class ResearchAgent:
    def __init__(self, model_type="gemini"):
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.llm    = get_gemini_llm() if model_type == "gemini" else get_groq_llm()

    def query(
        self,
        user_query: str,
        history:    List[Dict[str, Any]] = None,
    ) -> str:
        try:
            # ── Tavily search ─────────────────────────────
            search_results = self.tavily.search(user_query, max_results=3)

            if not search_results:
                return f"Research query failed: No search results for '{user_query}'"

            results = search_results.get("results", [])
            if not results:
                return f"Research query failed: No search results for '{user_query}'"

            # ── Extract content and reference links ───────
            combined_text   = ""
            reference_links = []

            for i, result in enumerate(results, start=1):
                title   = result.get("title",   "")
                content = result.get("content", "")
                url     = result.get("url",     "")

                if title or content:
                    combined_text += (
                        f"Title:   {title}\n"
                        f"URL:     {url}\n"
                        f"Content: {content}\n\n"
                    )

                if url:
                    reference_links.append(f"[{i}] {title}: {url}")

            if not combined_text.strip():
                return f"Could not extract content for query: {user_query}"

            # ── Build messages with history ───────────────
            messages = [
                SystemMessage(content=(
                    "You are a research assistant with web search capabilities.\n\n"

                    "Rules:\n"
                    "- Use the provided search results to answer the user's question.\n"
                    "- Do not invent facts that are not present in the search results.\n"
                    "- If information is missing, say it clearly.\n\n"

                    "Formatting Rules:\n"
                    "- Always respond using clean Markdown.\n"
                    "- Use headings (##) for major sections when helpful.\n"
                    "- Use bullet points instead of long paragraphs.\n"
                    "- Avoid large tables unless the user explicitly asks for them.\n"
                    "- Keep answers concise and readable for a chat interface.\n\n"

                    "Conversation Rules:\n"
                    "- You have access to the full conversation history.\n"
                    "- Use history to resolve follow-up questions.\n"
                    "- Maintain natural conversation flow.\n\n"

                    "Important:\n"
                    "- Do NOT include reference links inside the answer body.\n"
                    "- Reference links will be appended separately after the answer."
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

            # ✅ Add current query with search context
            messages.append(
                HumanMessage(content=(
                    f"Search Results:\n{combined_text[:4000]}\n\n"
                    f"Based on the above search results, provide a comprehensive "
                    f"answer to the query: '{user_query}'"
                ))
            )

            # ── Query LLM ─────────────────────────────────
            response = self.llm.invoke(messages)
            summary  = response.content

            if not summary:
                return f"Could not generate summary for query: {user_query}"

            # ── Append reference links (same as before) ───
            final_response = summary
            if reference_links:
                final_response += "\n\n--- Reference Links ---\n"
                for link in reference_links:
                    final_response += link + "\n"

            return final_response

        except Exception as e:
            error_msg = f"Research agent error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return error_msg