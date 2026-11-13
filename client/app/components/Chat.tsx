"use client";

import { useEffect, useRef, useState } from "react";

type Source = { source: string; heading: string; score: number };
type Msg = { role: "user" | "assistant"; text: string; sources?: Source[] };

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EXAMPLES = [
  "How many days of annual leave do I get?",
  "What is the home office setup stipend?",
  "What is the meal per diem when travelling?",
];

export default function Chat() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the newest message in view as the conversation grows.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, loading]);

  async function send(question: string) {
    const q = question.trim();
    if (!q || loading) return;

    setInput("");
    setError(null);
    setMsgs((m) => [...m, { role: "user", text: q }]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? `Request failed (${res.status})`);
      }

      const data = await res.json();
      setMsgs((m) => [
        ...m,
        { role: "assistant", text: data.answer, sources: data.sources },
      ]);
    } catch (e) {
      // Backend-down is the single most common failure for this app; make it legible.
      const msg =
        e instanceof TypeError
          ? "Cannot reach the backend. Is the server running?"
          : e instanceof Error
            ? e.message
            : "Something went wrong.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="messages">
        {msgs.length === 0 && !loading && (
          <div className="empty">
            <h2>Ask about company policy</h2>
            <p>Remote work, leave and time off, or expenses and travel.</p>
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  className="example"
                  onClick={() => send(ex)}
                  type="button"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {msgs.map((m, i) => (
          <div key={i} className={`row ${m.role}`}>
            {m.role === "assistant" && (
              <div className="avatar bot" aria-hidden>
                🧠
              </div>
            )}
            <div className="bubble">
              {m.text}
              {m.sources && m.sources.length > 0 && (
                <div className="sources">
                  <div className="sources-label">Sources</div>
                  {m.sources.map((s, j) => (
                    <span
                      key={j}
                      className="source-item"
                      title={`${s.heading} · similarity ${s.score.toFixed(2)}`}
                    >
                      {s.source}
                    </span>
                  ))}
                </div>
              )}
            </div>
            {m.role === "user" && (
              <div className="avatar user" aria-hidden>
                🧑
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="row assistant">
            <div className="avatar bot" aria-hidden>
              🧠
            </div>
            <div
              className="bubble typing"
              role="status"
              aria-label="WikiBrain is searching the policies"
            >
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
        {error && <div className="error">{error}</div>}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(input)}
          placeholder="Ask a policy question…"
          aria-label="Ask a policy question"
          disabled={loading}
        />
        <button onClick={() => send(input)} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </>
  );
}
