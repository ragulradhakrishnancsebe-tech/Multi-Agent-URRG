from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any
import os

# ── Embeddings ───────────────────────────────────────────
embedding_models = {
    "en":    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    "multi": HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
}


class RAGAgent:
    def __init__(self, model_type="gemini", embedding_type="en"):
        self.llm       = get_gemini_llm() if model_type == "gemini" else get_groq_llm()
        self.embedding = embedding_models[embedding_type]

    def query(
        self,
        user_query:    str,
        uploaded_docs: List[str],
        history:       List[Dict[str, Any]] = None,
    ) -> str:
        try:
            if not uploaded_docs:
                return "No documents uploaded for RAG query."

            # ── Load and split documents ─────────────────
            all_docs = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
            )

            for file_path in uploaded_docs:
                if os.path.exists(file_path):
                    try:
                        loader    = PyPDFLoader(file_path)
                        docs      = loader.load()
                        docs_split = splitter.split_documents(docs)
                        all_docs.extend(docs_split)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue

            if not all_docs:
                return "No readable content in uploaded documents."

            # ── Create Chroma vectorstore ─────────────────
            vectordb = Chroma.from_documents(all_docs, embedding=self.embedding)

            # ── Similarity search ─────────────────────────
            search_results = vectordb.similarity_search_with_score(user_query, k=3)
            context_text   = "\n\n".join([
                doc.page_content for doc, score in search_results
            ])

            # ── Build messages with history ───────────────
            messages = [
                SystemMessage(content=(
                    "You are a document analysis assistant.\n\n"

                    "Rules:\n"
                    "- Answer using ONLY the provided document context.\n"
                    "- If the answer is not present in the documents, clearly say: "
                    "'The answer is not found in the provided documents.'\n\n"

                    "Formatting Rules:\n"
                    "- Always respond in clean Markdown.\n"
                    "- Use headings (##) for main sections when appropriate.\n"
                    "- Use bullet points instead of long paragraphs.\n"
                    "- Avoid large tables unless the user explicitly asks for them.\n"
                    "- Keep responses concise and readable for chat UI.\n\n"

                    "Conversation Rules:\n"
                    "- You have access to the full conversation history.\n"
                    "- Use history to understand follow-up questions.\n"
                    "- Maintain a coherent conversation with the user."
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

            # ✅ Add current query with document context
            messages.append(
                HumanMessage(content=(
                    f"Document Context:\n{context_text}\n\n"
                    f"Question: {user_query}"
                ))
            )

            # ── Query LLM ─────────────────────────────────
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            error_msg = f"RAG agent error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return error_msg