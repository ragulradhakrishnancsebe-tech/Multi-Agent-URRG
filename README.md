# LangGraph Multi-Agent Chatbot

Full-stack AI chatbot using **FastAPI + LangGraph + React (Vite)**.

Supports:

* Multi-agent workflow
* RAG document search-PDF ONLY
* Real-time research agent
* Gemini / Groq model selection
* Session memory
* File uploads
* ChatGPT-style UI

---

# Project Structure

```
multi_agent_project/
│
├── agents/
│   ├── general_agent.py
│   ├── rag_agent.py
│   ├── research_agent.py
│   ├── summarize_agent.py
│   └── understanding_agent.py
│
├── models/
│   ├── gemini_model.py
│   └── groq_model.py
│
├── uploads/
│
├── .env
├── env_config.py
├── graph_flow.py
├── main.py
├── requirements.txt
│
├── UI/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── Chat.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── vite.config.js
│
└── .gitignore
```

---

# Backend Setup (FastAPI)

Install dependencies:

```
pip install -r requirements.txt
```

Run server:

```
uvicorn main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

Swagger docs:

```
http://127.0.0.1:8000/docs
```

---

# Frontend Setup (React + Vite)

Navigate to UI folder:

```
cd UI
```

Install dependencies:

```
npm install
```

Start frontend:

```
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

# API Endpoints

Upload file:

```
POST /upload
```

Chat with AI:

```
POST /chat
```

Reset session:

```
POST /reset
```

---

# Environment Variables

Create `.env` file:

```
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key
TAVILY_API_KEY=your_key
POSTGRES_URL=postgresql://user:password@localhost:5432/langgraph_db
```

---

# Tech Stack

Backend

* FastAPI
* LangGraph
* LangChain
* ChromaDB

Frontend

* React
* Vite
* Axios
* React Select
* Markdown rendering

AI Models

* Gemini
* Groq

---

# Author

Ragul
"# Multi-Agent-URRG" 
