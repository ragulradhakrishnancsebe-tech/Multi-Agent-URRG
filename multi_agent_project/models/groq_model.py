from langchain_groq import ChatGroq
from env_config import GROQ_API_KEY

def get_groq_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0,
        top_p=0.9,
        max_tokens=4096
    )