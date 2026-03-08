import { useState, useRef, useEffect } from "react";
import axios from "axios";
import Select from "react-select";
import "./Chat.css";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
// NovaMind
const modelOptions = [
  { value: "gemini", label: "✨ Gemini" },
  { value: "groq",   label: "⚡ Groq"   },
];

const parseMessageContent = function(content) {
  const refSplit = content.split(/---\s*Reference Links\s*---/i);
  if (refSplit.length < 2) return { body: content, links: [] };
  const body = refSplit[0].trim();
  const linkSection = refSplit[1];
  const links = [];
  const linkRegex = /\*?\s*([^:\n*]+?):\s*(https?:\/\/[^\s\n]+)/g;
  let match;
  while ((match = linkRegex.exec(linkSection)) !== null) {
    links.push({ label: match[1].trim(), url: match[2].trim() });
  }
  return { body: body, links: links };
};

const getOrCreateSessionId = function() {
  const existing = localStorage.getItem("chat_session_id");
  if (existing) return existing;
  const newId = "session_" + Date.now();
  localStorage.setItem("chat_session_id", newId);
  return newId;
};

const createNewSession = function() {
  const newId = "session_" + Date.now();
  localStorage.setItem("chat_session_id", newId);
  return newId;
};

function App() {
  const [query,       setQuery]       = useState("");
  const [messages,    setMessages]    = useState([]);
  const [sessionId,   setSessionId]   = useState(function() { return getOrCreateSessionId(); });
  const [model,       setModel]       = useState(modelOptions[0]);
  const [file,        setFile]        = useState(null);
  const [darkMode,    setDarkMode]    = useState(true);
  const [loading,     setLoading]     = useState(false);
  const [toast,       setToast]       = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const chatBoxRef = useRef(null);

  useEffect(function() {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages, loading]);

  useEffect(function() {
    if (!toast) return;
    const timer = setTimeout(function() { setToast(null); }, 3000);
    return function() { clearTimeout(timer); };
  }, [toast]);

  function showToast(msg, type) {
    setToast({ msg: msg, type: type || "success" });
  }

  async function handleFileUpload() {
    if (!file) {
      showToast("Select a file first", "error");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post(
        "http://localhost:8000/upload?session_id=" + sessionId,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      showToast(res.data.status || "File uploaded!");
      setFile(null);
    } catch (err) {
      console.error(err);
      showToast("Upload failed", "error");
    }
  }

  async function handleQuerySubmit() {
    if (!query.trim()) return;
    const userMsg = query;
    setMessages(function(prev) {
      return prev.concat([{ role: "user", content: userMsg }]);
    });
    setQuery("");
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/chat", {
        session_id: sessionId,
        query: userMsg,
        model: model.value,
      });
      const botResponse = res.data.response || "No response received";
      setMessages(function(prev) {
        return prev.concat([{ role: "bot", content: botResponse }]);
      });
    } catch (err) {
      console.error(err);
      setMessages(function(prev) {
        return prev.concat([{ role: "bot", content: "⚠️ Error fetching response" }]);
      });
    }
    setLoading(false);
  }

  async function handleReset() {
    try {
      await axios.post("http://localhost:8000/reset", { session_id: sessionId });
      setMessages([]);
      showToast("Memory cleared — session continues ✓");
    } catch (err) {
      console.error(err);
      showToast("Reset failed", "error");
    }
  }

  function handleNewSession() {
    const newId = createNewSession();
    setSessionId(newId);
    setMessages([]);
    setFile(null);
    showToast("New session started 🚀");
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") handleQuerySubmit();
  }

  const selectStyles = {
    control: function(base) {
      return Object.assign({}, base, {
        background:   darkMode ? "rgba(30,41,59,0.9)" : "rgba(255,255,255,0.9)",
        border:       "1px solid rgba(99,102,241,0.35)",
        borderRadius: "20px",
        boxShadow:    "none",
        minWidth:     "140px",
        cursor:       "pointer",
      });
    },
    menu: function(base) {
      return Object.assign({}, base, {
        background:   darkMode ? "#1e293b" : "#f8fafc",
        border:       "1px solid rgba(99,102,241,0.3)",
        borderRadius: "12px",
        overflow:     "hidden",
      });
    },
    option: function(base, state) {
      return Object.assign({}, base, {
        background: state.isFocused
          ? (darkMode ? "rgba(37,99,235,0.3)" : "rgba(37,99,235,0.1)")
          : "transparent",
        color:  darkMode ? "white" : "#1e293b",
        cursor: "pointer",
      });
    },
    singleValue: function(base) {
      return Object.assign({}, base, { color: darkMode ? "white" : "#1e293b" });
    },
    indicatorSeparator: function() { return { display: "none" }; },
    dropdownIndicator: function(base) {
      return Object.assign({}, base, { color: darkMode ? "#94a3b8" : "#64748b" });
    },
  };

  const wrapperClass   = "app-wrapper "    + (darkMode ? "dark" : "light");
  const sidebarClass   = "sidebar "        + (sidebarOpen ? "open " : "") + (darkMode ? "dark" : "light");
  const containerClass = "chat-container " + (darkMode ? "dark" : "light");
  const toastClass     = "toast "          + (toast && toast.type === "error" ? "toast-error" : "toast-success");

  return (
    <div className={wrapperClass}>

      <div className="bg-orb orb-1"></div>
      <div className="bg-orb orb-2"></div>
      <div className="bg-orb orb-3"></div>

      {toast ? (
        <div className={toastClass}>
          {toast.type === "error" ? "❌" : "✅"} {toast.msg}
        </div>
      ) : null}

      {sidebarOpen ? (
        <div className="sidebar-overlay" onClick={function() { setSidebarOpen(false); }} />
      ) : null}

      <aside className={sidebarClass}>

        <div className="sidebar-header">
          <div className="bot-avatar-lg">
            <span>🤖</span>
            <div className="online-dot"></div>
          </div>
          <div>
            <p className="sidebar-title">NovaMind</p>
            <p className="sidebar-sub">AI Assistant</p>
          </div>
        </div>

        <div className="sidebar-section">
          <p className="sidebar-label">Session</p>
          <div className="session-id-box">
            <span className="session-dot">●</span>
            <span className="session-id-text">{sessionId}</span>
          </div>
        </div>

        <div className="sidebar-section">
          <p className="sidebar-label">Model</p>
          <Select
            options={modelOptions}
            value={model}
            onChange={setModel}
            styles={selectStyles}
          />
        </div>

        <div className="sidebar-section">
          <p className="sidebar-label">Upload File</p>
          <label className="file-label">
            <span>📎 {file ? file.name.slice(0, 16) + "…" : "Choose File"}</span>
            <input
              type="file"
              onChange={function(e) { setFile(e.target.files[0]); }}
              hidden
            />
          </label>
          <button className="btn-upload" onClick={handleFileUpload}>
            ⬆ Upload
          </button>
        </div>

        <div className="sidebar-section">
          <p className="sidebar-label">Session Actions</p>
          <button className="btn-reset" onClick={handleReset}>
            🧹 Clear Memory
          </button>
          <button className="btn-new-session" onClick={handleNewSession}>
            ✨ New Chat
          </button>
        </div>

        <div className="sidebar-footer">
          <button className="mode-toggle" onClick={function() { setDarkMode(!darkMode); }}>
            {darkMode ? "☀️ Light Mode" : "🌙 Dark Mode"}
          </button>
        </div>

      </aside>

      <main className={containerClass}>

        <header>
          <button className="hamburger" onClick={function() { setSidebarOpen(!sidebarOpen); }}>
            ☰
          </button>

          <div className="header-center">
            <div className="bot-avatar">
              <span>🤖</span>
              <div className="online-dot"></div>
            </div>
            <div>
              <h1>NovaMind Chatbot</h1>
              <span className="header-status">● Online</span>
            </div>
          </div>

          <div className="header-right">
            <div className="inline-controls">
              <Select
                options={modelOptions}
                value={model}
                onChange={setModel}
                styles={selectStyles}
              />
              <label className="file-label">
                <span>📎 {file ? file.name.slice(0, 10) + "…" : "File"}</span>
                <input
                  type="file"
                  onChange={function(e) { setFile(e.target.files[0]); }}
                  hidden
                />
              </label>
              <button className="btn-upload sm" onClick={handleFileUpload}>⬆</button>
              <button className="btn-reset sm" onClick={handleReset}>🧹</button>
              <button className="btn-new-session sm" onClick={handleNewSession}>✨</button>
            </div>
            <button className="mode-toggle" onClick={function() { setDarkMode(!darkMode); }}>
              {darkMode ? "☀️" : "🌙"}
            </button>
          </div>
        </header>

        <div className="chat-box" ref={chatBoxRef}>

          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">💬</div>
              <p>Ask me anything or upload a file to begin!</p>
              <div className="session-pill">
                <span>🔑</span>
                <span>{sessionId}</span>
              </div>
            </div>
          ) : null}

          {messages.map(function(msg, index) {
            const parsed = msg.role === "bot"
              ? parseMessageContent(msg.content)
              : null;

            return (
              <div key={index} className={"message " + msg.role}>

                {msg.role === "bot" ? (
                  <div className="msg-avatar bot-av">🤖</div>
                ) : null}

                <div className="bubble">
                  {msg.role === "bot" ? (
                    <div>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {parsed.body}
                      </ReactMarkdown>

                      {parsed.links.length > 0 ? (
                        <div className="ref-links-box">
                          <div className="ref-links-title">
                            <span className="ref-icon">🔗</span>
                            <span> Sources</span>
                          </div>
                          <div className="ref-links-list">
                            {parsed.links.map(function(link, i) {
                              return (
                                <a
                                  key={i}
                                  href={link.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="ref-link-item"
                                >
                                  <span className="ref-link-num">{i + 1}</span>
                                  <span className="ref-link-label">{link.label}</span>
                                  <span className="ref-link-arrow">↗</span>
                                </a>
                              );
                            })}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <span>{msg.content}</span>
                  )}
                </div>

                {msg.role === "user" ? (
                  <div className="msg-avatar user-av">🧑</div>
                ) : null}

              </div>
            );
          })}

          {loading ? (
            <div className="message bot">
              <div className="msg-avatar bot-av">🤖</div>
              <div className="bubble typing-bubble">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          ) : null}

        </div>

        <div className="input-area">
          <input
            type="text"
            value={query}
            onChange={function(e) { setQuery(e.target.value); }}
            placeholder="Type your question..."
            onKeyDown={handleKeyDown}
          />
          <button onClick={handleQuerySubmit} disabled={loading}>
            {loading ? "⏳" : "➤"}
          </button>
        </div>

      </main>
    </div>
  );
}

export default App;