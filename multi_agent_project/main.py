from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# --- Safe imports ---
try:
    from graph_flow import build_graph
except Exception as e:
    raise ImportError(f"Error importing graph_flow: {e}")

try:
    from env_config import POSTGRES_URL, UPLOAD_FOLDER
except Exception:
    POSTGRES_URL  = ""
    UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI(title="Multi-Agent LangGraph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---
class QueryRequest(BaseModel):
    session_id: str
    query:      str
    model:      str = "gemini"

class SessionRequest(BaseModel):
    session_id: str

# ✅ Single source of truth for memory and docs
SESSION_MEMORY: dict = {}   # { session_id: [ {user: ..., assistant: ...} ] }
SESSION_DOCS:   dict = {}   # { session_id: [ file_path, ... ] }

# ✅ Store conversation summary per session
SESSION_SUMMARY: dict = {}  # { session_id: "conversation summary" }

# --- Graph cache per model ---
GRAPH_CACHE: dict = {}

# ✅ Limit memory size to prevent token explosion
MAX_HISTORY = 4


def trim_history(history):
    if not history:
        return []
    return history[-MAX_HISTORY:]


def get_compiled_graph(model_type: str):
    """Cache compiled graph — never rebuild unless model changes."""
    if model_type not in GRAPH_CACHE:
        print(f"🔧 Building graph for model: {model_type}")
        GRAPH_CACHE[model_type] = build_graph(model_type)
    return GRAPH_CACHE[model_type]


# --- Upload endpoint ---
@app.post("/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    try:
        file_path = f"{UPLOAD_FOLDER}/{session_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        SESSION_DOCS.setdefault(session_id, []).append(file_path)
        return {"status": f"✅ '{file.filename}' uploaded successfully"}
    except Exception as e:
        return {"status": f"❌ Upload failed: {str(e)}"}


# --- Reset endpoint ---
@app.post("/reset")
async def reset_session(request: SessionRequest):
    # ✅ Clear memory and docs for this session only
    SESSION_MEMORY.pop(request.session_id, None)
    SESSION_DOCS.pop(request.session_id,   None)
    SESSION_SUMMARY.pop(request.session_id, None)

    print(f"🔄 Session '{request.session_id}' cleared")
    return {"status": "session cleared successfully"}


# --- Chat endpoint ---
@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        model_type     = request.model or "gemini"
        compiled_graph = get_compiled_graph(model_type)

        # ✅ Get history from SESSION_MEMORY
        history = SESSION_MEMORY.get(request.session_id, [])

        # ✅ Trim history to reduce tokens
        history = trim_history(history)

        print(f"📋 Session '{request.session_id}' | "
              f"turns: {len(history)} | query: '{request.query}'")

        # --- Build state ---
        initial_state = {
            "query":             request.query,
            "history":           history,
            "uploaded_docs":     SESSION_DOCS.get(request.session_id, []),
            "category":          "",
            "rag_response":      "",
            "research_response": "",
            "general_response":  "",
            "summary":           SESSION_SUMMARY.get(request.session_id, ""),  # ✅ pass summary
        }

        # ✅ Config required because graph compiled with checkpointer
        config = {
            "configurable": {
                "thread_id": request.session_id
            }
        }

        # --- Invoke graph with config ---
        final_state = compiled_graph.invoke(initial_state, config=config)

        # ✅ Save updated summary
        if final_state.get("summary"):
            SESSION_SUMMARY[request.session_id] = final_state["summary"]

        # --- Extract best response ---
        response = None
        for key in ["summary", "research_response", "rag_response", "general_response"]:
            val = final_state.get(key, "")
            if val and isinstance(val, str) and val.strip():
                response = val.strip()
                break

        # --- Fallback agent ---
        if not response:
            from agents.general_agent import GeneralAgent
            fallback = GeneralAgent(model_type).respond(request.query, history)
            if fallback and isinstance(fallback, str) and fallback.strip():
                response = fallback.strip()

        if not response:
            response = "Unable to generate a response. Please try again."

        # ✅ Save this turn into SESSION_MEMORY
        SESSION_MEMORY.setdefault(request.session_id, []).append({
            "user":      request.query,
            "assistant": response,
        })

        # ✅ Trim stored memory as well
        SESSION_MEMORY[request.session_id] = trim_history(
            SESSION_MEMORY[request.session_id]
        )

        print(f"✅ Turn {len(SESSION_MEMORY[request.session_id])} saved "
              f"— session '{request.session_id}'")

        return {"response": response, "session_id": request.session_id}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "response": f"Error: {str(e)}",
            "error":    str(e),
        }