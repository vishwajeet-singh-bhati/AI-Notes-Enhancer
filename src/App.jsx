import { useState, useRef } from "react";

const SAMPLE_TEXT = `I am doing project on AI which is very good and it help many people in future. the system use machine learning and deep learning for make prediction. I think this technology is very important because it save time and reduce error in many field like healthcare and education and also business. we should use this more.`;

function cosineSimilarity(a, b) {
  // Simple TF-based cosine similarity for client-side scoring
  const tokenize = (text) =>
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .split(/\s+/)
      .filter(Boolean);

  const tokA = tokenize(a);
  const tokB = tokenize(b);
  const vocab = [...new Set([...tokA, ...tokB])];

  const vec = (toks) => vocab.map((w) => toks.filter((t) => t === w).length);
  const vA = vec(tokA);
  const vB = vec(tokB);

  const dot = vA.reduce((s, v, i) => s + v * vB[i], 0);
  const magA = Math.sqrt(vA.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(vB.reduce((s, v) => s + v * v, 0));
  return magA && magB ? dot / (magA * magB) : 0;
}

function diffWords(original, enhanced) {
  // Simple word-level diff to highlight changes
  const oWords = original.split(/(\s+)/);
  const eWords = enhanced.split(/(\s+)/);
  return { original: oWords, enhanced: eWords };
}

function ChangeLogItem({ index, change }) {
  return (
    <div className="changelog-item" style={{ animationDelay: `${index * 60}ms` }}>
      <div className="change-number">{String(index + 1).padStart(2, "0")}</div>
      <div className="change-content">
        <div className="change-pair">
          <span className="change-original">"{change.original}"</span>
          <span className="change-arrow">→</span>
          <span className="change-enhanced">"{change.enhanced}"</span>
        </div>
        <div className="change-reason">{change.reason}</div>
      </div>
    </div>
  );
}

function ScoreRing({ score }) {
  const pct = Math.round(score * 100);
  const r = 38;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  const color = pct >= 90 ? "#4ade80" : pct >= 75 ? "#facc15" : "#f87171";

  return (
    <div className="score-ring-wrap">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={r} fill="none" stroke="#1e293b" strokeWidth="8" />
        <circle
          cx="50"
          cy="50"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
          style={{ transition: "stroke-dasharray 1s ease" }}
        />
        <text x="50" y="54" textAnchor="middle" fill={color} fontSize="18" fontWeight="700" fontFamily="'DM Mono', monospace">
          {pct}%
        </text>
      </svg>
      <div className="score-label">Semantic Similarity</div>
    </div>
  );
}

export default function App() {
  const [input, setText] = useState(SAMPLE_TEXT);
  const [output, setOutput] = useState("");
  const [changes, setChanges] = useState([]);
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [phase, setPhase] = useState("idle"); // idle | enhancing | scoring | done

  const enhance = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError("");
    setOutput("");
    setChanges([]);
    setScore(null);
    setPhase("enhancing");

    try {
      const systemPrompt = `You are a professional writing editor specializing in clear, polished, institutional-quality English. Your job is to improve grammar, fix awkward phrasing, eliminate colloquialisms, and elevate the tone to professional standard — while keeping every original idea and fact intact.

Respond with ONLY valid JSON (no markdown, no backticks) in this exact structure:
{
  "enhanced": "the improved text",
  "changes": [
    { "original": "original phrase", "enhanced": "corrected phrase", "reason": "brief reason" }
  ]
}

GRAMMAR & STRUCTURE RULES:
- Fix all grammar errors: tense, subject-verb agreement, missing articles, wrong prepositions.
- Fix spelling and punctuation mistakes.
- Split run-on sentences into clear, well-punctuated sentences.
- Capitalize proper nouns and the start of sentences correctly.

TONE ELEVATION RULES — apply these always:
- Expand ALL contractions: "don't" → "do not", "they're" → "they are", "won't" → "will not", "can't" → "cannot", "it's" → "it is", "we'll" → "we will", "I'm" → "I am", etc.
- Replace colloquial Indian-English phrases with professional equivalents:
  * "try our level best" → "make every effort" / "take all necessary steps"
  * "do the needful" → "take the required action"
  * "revert back" → "respond"
  * "prepone" → "reschedule to an earlier date"
  * "good name" → "name"
  * "kindly do" → "please" or restructure
- Replace vague or weak words with precise professional ones:
  * "problematic" → "challenging" / "a concern" / "difficult to manage"
  * "too busy" → "occupied" / "otherwise engaged"
  * "somewhat difficult" → "challenging" / "requires careful attention"
  * "problem can happen" → "incidents may occur" / "complications may arise"
  * "water should be there" → "water should be readily available"
  * "stand there for watching" → "remain present for supervision"
  * "no harmful colors allowed" → "the use of harmful colors is strictly prohibited"
  * "should not go near fire" → "should maintain a safe distance from the fire"
  * "things needed" → "required materials" / "necessary items"
  * "and also" (repeated) → restructure with proper conjunctions or separate sentences

CLARITY & FLUENCY RULES:
- Make sentences flow naturally — like a confident, clear, professional native English speaker.
- Replace vague words with precise natural equivalents where meaning is obvious from context.
- Do not make it sound like a legal document — professional but readable.
- Avoid redundant subject repetition: "they said X and they also said Y" → "they said X and also Y" / restructure to eliminate the repeated pronoun.
- Avoid stacking transition words back-to-back: never use "Nevertheless" and "However" (or similar: "Moreover", "Furthermore", "Additionally") within the same sentence or consecutive sentences. Vary transitions or restructure instead.
- Remove weak hedge words that dilute professional tone:
  * "somewhat" → remove entirely or replace ("somewhat challenging" → "challenging", "somewhat difficult" → "difficult")
  * "kind of" → remove
  * "sort of" → remove
  * "a bit" → remove or replace ("a bit complex" → "complex")
  * "quite" → use sparingly, remove if weakening
- Avoid AI-ish phrasing that sounds mechanical: do not produce outputs that feel like a list of transitions strung together.
- Prefer active constructions where they sound more natural than passive.

HARD CONSTRAINTS:
- Do NOT add any new facts, figures, names, or ideas not present in the original.
- Do NOT remove any information or idea from the original.
- Do NOT change what the person means — only how they say it.
- Each change log entry must show the specific original phrase and the exact replacement with a clear reason.`;

      const apiKey = import.meta.env.VITE_ANTHROPIC_API_KEY;
      if (!apiKey) throw new Error("API key not configured. Add VITE_ANTHROPIC_API_KEY to your .env file.");
      const callAPI = async (attempt = 1) => {
        const r = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
            "x-api-key": apiKey,
          },
          body: JSON.stringify({
            model: "claude-haiku-4-5-20251001",
            max_tokens: 1500,
            system: systemPrompt,
            messages: [{ role: "user", content: `Enhance this text:\n\n${input}` }],
          }),
        });
        const d = await r.json();
        if (d?.error?.type === "overloaded_error" && attempt < 4) {
          await new Promise(res => setTimeout(res, attempt * 1500));
          return callAPI(attempt + 1);
        }
        if (!r.ok) throw new Error(d?.error?.message || "HTTP " + r.status);
        return d;
      };

      const data = await callAPI();
      const raw = data.content.map((b) => b.text || "").join("");
      const clean = raw.replace(/```json\n?|```/g, "").trim();
      const parsed = JSON.parse(clean);

      setOutput(parsed.enhanced || "");
      setChanges(parsed.changes || []);

      setPhase("scoring");
      await new Promise((r) => setTimeout(r, 600));
      const sim = cosineSimilarity(input, parsed.enhanced || "");
      setScore(sim);
      setPhase("done");
    } catch (e) {
      setError("Error: " + (e.message || String(e)));
      setPhase("idle");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setOutput("");
    setChanges([]);
    setScore(null);
    setPhase("idle");
    setError("");
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #050a14;
          --surface: #0c1628;
          --surface2: #111f36;
          --border: #1e3a5f;
          --accent: #38bdf8;
          --accent2: #818cf8;
          --green: #4ade80;
          --red: #f87171;
          --text: #e2e8f0;
          --muted: #64748b;
          --glow: 0 0 30px rgba(56,189,248,0.15);
        }

        body {
          background: var(--bg);
          color: var(--text);
          font-family: 'DM Sans', sans-serif;
          min-height: 100vh;
          overflow-x: hidden;
        }

        .app {
          max-width: 1300px;
          margin: 0 auto;
          padding: 32px 24px 80px;
        }

        .header {
          text-align: center;
          padding: 48px 0 40px;
          position: relative;
        }

        .header-tag {
          display: inline-block;
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          letter-spacing: 0.2em;
          color: var(--accent);
          background: rgba(56,189,248,0.08);
          border: 1px solid rgba(56,189,248,0.2);
          padding: 4px 14px;
          border-radius: 20px;
          margin-bottom: 20px;
          text-transform: uppercase;
        }

        .header h1 {
          font-family: 'Syne', sans-serif;
          font-size: clamp(2rem, 5vw, 3.6rem);
          font-weight: 800;
          line-height: 1.1;
          background: linear-gradient(135deg, #e2e8f0 0%, var(--accent) 50%, var(--accent2) 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-bottom: 12px;
        }

        .header p {
          color: var(--muted);
          font-size: 15px;
          max-width: 520px;
          margin: 0 auto;
          line-height: 1.6;
        }

        .editor-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          margin-bottom: 24px;
        }

        @media (max-width: 768px) {
          .editor-grid { grid-template-columns: 1fr; }
        }

        .panel {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .panel-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 14px 20px;
          border-bottom: 1px solid var(--border);
          background: var(--surface2);
        }

        .panel-title {
          font-family: 'Syne', sans-serif;
          font-size: 13px;
          font-weight: 600;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          color: var(--muted);
        }

        .panel-badge {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          padding: 2px 10px;
          border-radius: 10px;
          letter-spacing: 0.1em;
        }

        .badge-original { background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }
        .badge-enhanced { background: rgba(74,222,128,0.1); color: #4ade80; border: 1px solid rgba(74,222,128,0.2); }

        textarea {
          flex: 1;
          background: transparent;
          border: none;
          outline: none;
          color: var(--text);
          font-family: 'DM Sans', sans-serif;
          font-size: 15px;
          line-height: 1.7;
          padding: 20px;
          resize: none;
          min-height: 260px;
        }

        textarea::placeholder { color: var(--muted); }

        .output-text {
          padding: 20px;
          font-size: 15px;
          line-height: 1.7;
          min-height: 260px;
          white-space: pre-wrap;
          flex: 1;
        }

        .output-placeholder {
          color: var(--muted);
          font-style: italic;
          display: flex;
          align-items: center;
          justify-content: center;
          height: 260px;
          flex-direction: column;
          gap: 12px;
        }

        .placeholder-icon {
          font-size: 36px;
          opacity: 0.3;
        }

        .action-bar {
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 28px;
          flex-wrap: wrap;
        }

        .btn-enhance {
          font-family: 'Syne', sans-serif;
          font-weight: 700;
          font-size: 15px;
          letter-spacing: 0.03em;
          color: #050a14;
          background: linear-gradient(135deg, var(--accent), var(--accent2));
          border: none;
          padding: 14px 40px;
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.2s;
          box-shadow: 0 0 24px rgba(56,189,248,0.3);
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .btn-enhance:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 0 40px rgba(56,189,248,0.5);
        }

        .btn-enhance:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .btn-reset {
          font-family: 'DM Sans', sans-serif;
          font-size: 14px;
          color: var(--muted);
          background: transparent;
          border: 1px solid var(--border);
          padding: 13px 24px;
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-reset:hover { color: var(--text); border-color: var(--accent); }

        .phase-indicator {
          font-family: 'DM Mono', monospace;
          font-size: 12px;
          color: var(--accent);
          display: flex;
          align-items: center;
          gap: 8px;
          animation: pulse 1.5s ease infinite;
        }

        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

        .dot-loader {
          display: flex;
          gap: 4px;
        }

        .dot-loader span {
          width: 5px; height: 5px;
          background: var(--accent);
          border-radius: 50%;
          animation: dotBounce 1.2s ease infinite;
        }

        .dot-loader span:nth-child(2) { animation-delay: 0.2s; }
        .dot-loader span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes dotBounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }

        .results-section {
          animation: fadeUp 0.5s ease;
        }

        @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

        .results-header {
          display: flex;
          align-items: center;
          gap: 20px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 24px 28px;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }

        .score-ring-wrap {
          text-align: center;
          flex-shrink: 0;
        }

        .score-label {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          color: var(--muted);
          letter-spacing: 0.1em;
          text-transform: uppercase;
          margin-top: 4px;
        }

        .score-info {
          flex: 1;
        }

        .score-info h3 {
          font-family: 'Syne', sans-serif;
          font-size: 18px;
          font-weight: 700;
          margin-bottom: 6px;
        }

        .score-info p {
          font-size: 13px;
          color: var(--muted);
          line-height: 1.6;
        }

        .stats-row {
          display: flex;
          gap: 20px;
          flex-wrap: wrap;
        }

        .stat-chip {
          background: var(--surface2);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 10px 16px;
          font-family: 'DM Mono', monospace;
        }

        .stat-chip .stat-val {
          font-size: 20px;
          font-weight: 500;
          color: var(--accent);
        }

        .stat-chip .stat-name {
          font-size: 10px;
          color: var(--muted);
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .changelog-wrap {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          overflow: hidden;
        }

        .changelog-header {
          padding: 16px 24px;
          border-bottom: 1px solid var(--border);
          background: var(--surface2);
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .changelog-header h3 {
          font-family: 'Syne', sans-serif;
          font-size: 14px;
          font-weight: 700;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .count-badge {
          background: rgba(56,189,248,0.1);
          border: 1px solid rgba(56,189,248,0.2);
          color: var(--accent);
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          padding: 1px 8px;
          border-radius: 20px;
        }

        .changelog-list {
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .changelog-item {
          display: flex;
          gap: 14px;
          background: var(--surface2);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 14px 16px;
          animation: fadeUp 0.4s ease both;
          transition: border-color 0.2s;
        }

        .changelog-item:hover { border-color: var(--accent); }

        .change-number {
          font-family: 'DM Mono', monospace;
          font-size: 12px;
          color: var(--accent);
          opacity: 0.6;
          min-width: 24px;
          padding-top: 2px;
        }

        .change-content { flex: 1; }

        .change-pair {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
          margin-bottom: 6px;
        }

        .change-original {
          font-family: 'DM Mono', monospace;
          font-size: 12px;
          color: var(--red);
          background: rgba(248,113,113,0.08);
          padding: 2px 8px;
          border-radius: 6px;
          text-decoration: line-through;
          word-break: break-word;
        }

        .change-enhanced {
          font-family: 'DM Mono', monospace;
          font-size: 12px;
          color: var(--green);
          background: rgba(74,222,128,0.08);
          padding: 2px 8px;
          border-radius: 6px;
          word-break: break-word;
        }

        .change-arrow { color: var(--muted); font-size: 14px; }

        .change-reason {
          font-size: 12px;
          color: var(--muted);
          line-height: 1.5;
        }

        .error-msg {
          background: rgba(248,113,113,0.08);
          border: 1px solid rgba(248,113,113,0.2);
          color: var(--red);
          padding: 12px 18px;
          border-radius: 10px;
          font-size: 14px;
        }

        .char-count {
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          color: var(--muted);
        }

        .spinner {
          width: 16px; height: 16px;
          border: 2px solid rgba(5,10,20,0.3);
          border-top-color: #050a14;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .gradient-line {
          height: 1px;
          background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent);
          margin: 32px 0;
          opacity: 0.3;
        }
      `}</style>

      <div className="app">
        <div className="header">
          <div className="header-tag">✦ AI Writing Assistant</div>
          <h1>Meaning-Preserving<br />Notes Enhancer</h1>
          <p>Improve your writing without changing what you mean. Grammar, clarity, and structure — never your ideas.</p>
        </div>

        <div className="editor-grid">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Original Text</span>
              <span className="char-count">{input.length} chars</span>
            </div>
            <textarea
              value={input}
              onChange={(e) => { setText(e.target.value); reset(); }}
              placeholder="Paste or type your raw, unpolished text here..."
            />
          </div>

          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Enhanced Text</span>
              {output && <span className="panel-badge badge-enhanced">✓ READY</span>}
            </div>
            {output ? (
              <div className="output-text">{output}</div>
            ) : (
              <div className="output-text">
                <div className="output-placeholder">
                  <span className="placeholder-icon">✦</span>
                  <span>Enhanced text will appear here</span>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="action-bar">
          <button className="btn-enhance" onClick={enhance} disabled={loading || !input.trim()}>
            {loading ? <div className="spinner" /> : "✦"}
            {loading
              ? phase === "enhancing" ? "Enhancing..." : "Scoring..."
              : "Enhance Writing"}
          </button>

          {output && (
            <button className="btn-reset" onClick={reset}>Reset</button>
          )}

          {loading && (
            <div className="phase-indicator">
              <div className="dot-loader">
                <span /><span /><span />
              </div>
              {phase === "enhancing" ? "Analyzing and improving expression... (retries automatically if busy)" : "Computing semantic similarity..."}
            </div>
          )}

          {error && <div className="error-msg">⚠ {error}</div>}
        </div>

        {score !== null && output && (
          <div className="results-section">
            <div className="results-header">
              <ScoreRing score={score} />
              <div className="score-info">
                <h3>
                  {score >= 0.92 ? "🟢 Meaning Fully Preserved" : score >= 0.78 ? "🟡 Meaning Largely Preserved" : "🔴 Meaning May Have Shifted"}
                </h3>
                <p>
                  {score >= 0.92
                    ? "The enhanced text maintains strong semantic alignment with your original. Only expression was improved."
                    : score >= 0.78
                    ? "The enhanced text is close in meaning. Review the changes carefully to confirm intent."
                    : "Significant divergence detected. Consider reviewing the enhanced text closely."}
                </p>
                <div style={{ marginTop: 14 }} className="stats-row">
                  <div className="stat-chip">
                    <div className="stat-val">{changes.length}</div>
                    <div className="stat-name">Changes Made</div>
                  </div>
                  <div className="stat-chip">
                    <div className="stat-val">{Math.round(score * 100)}%</div>
                    <div className="stat-name">Similarity</div>
                  </div>
                  <div className="stat-chip">
                    <div className="stat-val">{output.split(/\s+/).length}</div>
                    <div className="stat-name">Word Count</div>
                  </div>
                </div>
              </div>
            </div>

            {changes.length > 0 && (
              <div className="changelog-wrap">
                <div className="changelog-header">
                  <h3>Change Log</h3>
                  <span className="count-badge">{changes.length} modifications</span>
                </div>
                <div className="changelog-list">
                  {changes.map((c, i) => (
                    <ChangeLogItem key={i} index={i} change={c} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}
