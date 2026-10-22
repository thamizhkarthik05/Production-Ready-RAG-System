
import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    const controller = new AbortController();

    async function checkBackend() {
      try {
        const response = await fetch("/api/health", {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Backend health check failed");
        }

        const data = await response.json();

        setApiStatus(data.status === "ok" ? "connected" : "error");
      } catch (error) {
        if (error.name !== "AbortError") {
          setApiStatus("error");
        }
      }
    }

    checkBackend();

    return () => controller.abort();
  }, []);

  const statusText = {
    checking: "Connecting to backend...",
    connected: "Backend connected",
    error: "Backend unavailable",
  };

  return (
    <main className="app-shell">
      <nav className="navbar">
        <a className="brand" href="/" aria-label="Ask My Doc home">
          <span className="brand-mark">A</span>
          <span>Ask My Doc</span>
        </a>

        <span className={`status status-${apiStatus}`}>
          <span className="status-dot" />
          {statusText[apiStatus]}
        </span>
      </nav>

      <section className="hero">
        <div className="eyebrow">
          <span className="eyebrow-line" />
          YOUR DOCUMENTS. YOUR ANSWERS.
        </div>

        <h1>
          Knowledge,
          <br />
          <span>within reach.</span>
        </h1>

        <p className="hero-description">
          A smarter way to explore your documents.
          Ask questions, discover relevant evidence, and
          get answers grounded in your own sources.
        </p>

        <div className="feature-list">
          <div className="feature">
            <span className="feature-icon">01</span>
            <div>
              <strong>Evidence first</strong>
              <p>Answers supported by source passages.</p>
            </div>
          </div>

          <div className="feature">
            <span className="feature-icon">02</span>
            <div>
              <strong>Built for precision</strong>
              <p>Retrieval designed to find relevant context.</p>
            </div>
          </div>

          <div className="feature">
            <span className="feature-icon">03</span>
            <div>
              <strong>Transparent answers</strong>
              <p>Trace responses back to their sources.</p>
            </div>
          </div>
        </div>

        <div className="setup-notice">
          <span className="notice-indicator" />
          <div>
            <strong>Foundation is ready</strong>
            <p>
              Document ingestion and question answering
              will be added in the next development steps.
            </p>
          </div>
        </div>
      </section>

      <footer className="footer">
        <span>ASK MY DOC</span>
        <span>Evidence-aware RAG · v0.1.0</span>
      </footer>
    </main>
  );
}

export default App;
