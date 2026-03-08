import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")