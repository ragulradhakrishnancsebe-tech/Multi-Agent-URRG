from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.gemini_model import get_gemini_llm
from models.groq_model import get_groq_llm
from typing import List, Dict, Any
import os


embedding_models = {
    "en": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    "multi": HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
}


class RAGAgent:

    def __init__(self, model_type="gemini", embedding_type="en"):

        self.llm = get_gemini_llm() if model_type == "gemini" else get_groq_llm()
        self.embedding = embedding_models.get(embedding_type, embedding_models["en"])

        self.vectordb = None
        self.loaded_docs = set()


    # ── Build vector database once ─────────────────

    def load_documents(self, uploaded_docs: List[str]):

        all_docs = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )

        for file_path in uploaded_docs:

            if file_path in self.loaded_docs:
                continue

            if os.path.exists(file_path):

                loader = PyPDFLoader(file_path)
                docs = loader.load()

                docs_split = splitter.split_documents(docs)
                all_docs.extend(docs_split)

                self.loaded_docs.add(file_path)

        if not all_docs:
            return False

        if self.vectordb is None:
            self.vectordb = Chroma.from_documents(
                all_docs,
                embedding=self.embedding
            )
        else:
            self.vectordb.add_documents(all_docs)

        return True


    # ── Query documents ─────────────────────────────

    def query(
        self,
        user_query: str,
        history: List[Dict[str, Any]] = None,
    ) -> str:

        try:

            if self.vectordb is None:
                return "Documents not loaded."

            search_results = self.vectordb.similarity_search(
                user_query,
                k=2
            )

            if not search_results:
                return "No relevant information found in uploaded documents."

            context_text = "\n\n".join(
                [doc.page_content for doc in search_results]
            )

            messages = [
                SystemMessage(
                    content=(
                        "Answer using the provided document context. "
                        "If the answer is not in the document, say so. "
                        "Respond clearly using markdown, short sections, and bullet points."
                    )
                )
            ]

            # limit history
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

            messages.append(
                HumanMessage(
                    content=f"Context:\n{context_text}\n\nQuestion: {user_query}"
                )
            )

            response = self.llm.invoke(messages)

            if hasattr(response, "content"):
                return response.content

            return str(response)

        except Exception as e:

            error_msg = f"RAG agent error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return error_msg