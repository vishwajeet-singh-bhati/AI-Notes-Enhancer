import { useState, useRef, useCallback, useReducer } from "react";

// ─── Constants ───────────────────────────────────────────────────────────────

const MAX_CHARS = 4000;

const SAMPLE_TEXT = `I am doing project on AI which is very good and it help many people in future. the system use machine learning and deep learning for make prediction. I think this technology is very important because it save time and reduce error in many field like healthcare and education and also business. we should use this more.`;

const TONE_MODES = [
  {
    id: "casual",
    label: "Casual Fix",
    icon: "💬",
    desc: "Fix grammar only, keep informal tone",
    color: "#38bdf8",
    prompt: `You are a grammar-only editor. Fix grammar, spelling, and punctuation ONLY. Do NOT change tone, vocabulary, or style. Keep it informal and natural.

Respond with ONLY valid JSON:
{"enhanced": "fixed text", "changes": [{"original": "phrase", "enhanced": "fix", "reason": "why"}]}

RULES:
- Fix grammar errors, spelling mistakes, punctuation only.
- Keep casual/informal words exactly as they are.
- Do NOT expand contractions.
- Do NOT replace simple words with formal ones.
- Do NOT change the tone at all.`,
  },
  {
    id: "professional",
    label: "Professional",
    icon: "💼",
    desc: "Elevate to professional tone",
    color: "#818cf8",
    prompt: `You are a professional writing editor. Improve grammar, fix awkward phrasing, eliminate colloquialisms, and elevate tone to professional standard — while keeping every original idea intact.

Respond with ONLY valid JSON:
{"enhanced": "improved text", "changes": [{"original": "phrase", "enhanced": "replacement", "reason": "why"}]}

RULES:
- Fix all grammar errors, spelling, punctuation, run-on sentences.
- Expand ALL contractions: don't→do not, can't→cannot, they're→they are.
- Replace colloquial phrases: "try our level best"→"make every effort", "do the needful"→"take the required action".
- Remove hedge words: "somewhat", "kind of", "sort of".
- Do NOT add new facts or remove existing ones.`,
  },
  {
    id: "academic",
    label: "Academic",
    icon: "🎓",
    desc: "Scholarly tone for reports & research",
    color: "#4ade80",
    prompt: `You are an academic writing editor. Transform text into scholarly, precise academic English suitable for research papers and reports.

Respond with ONLY valid JSON:
{"enhanced": "improved text", "changes": [{"original": "phrase", "enhanced": "replacement", "reason": "why"}]}

RULES:
- Use formal academic vocabulary and sentence structures.
- Expand ALL contractions without exception.
- Replace "I think"→"It can be argued that", "shows"→"demonstrates", "a lot of"→"a significant number of".
- Use passive voice where academically appropriate.
- Do NOT add new facts or arguments not in the original.`,
  },
  {
    id: "email",
    label: "Email",
    icon: "📧",
    desc: "Polished email communication",
    color: "#facc15",
    prompt: `You are an email writing editor. Transform rough notes into clear, professional email communication.

Respond with ONLY valid JSON:
{"enhanced": "improved text", "changes": [{"original": "phrase", "enhanced": "replacement", "reason": "why"}]}

RULES:
- Expand contractions: don't→do not, I'm→I am.
- Replace casual phrases: "can you"→"could you please", "ASAP"→"at your earliest convenience".
- Remove filler words: "just", "basically", "kind of".
- Keep sentences concise and polite.
- Do NOT add new information not in the original.`,
  },
];

// ─── Embedding Cache ──────────────────────────────────────────────────────────
// Caches embeddings by text to avoid re-fetching for the same input
const embeddingCache = new Map();

// ─── API Helpers ──────────────────────────────────────────────────────────────

/**
 * FIX: API keys should be server-side in production.
 * For local dev, keys are read from env vars — never commit them.
 * In production, replace fetch URLs with your own backend proxy endpoints:
 *   /api/embed   → calls Gemini
 *   /api/enhance → calls Groq
 */
const getEmbedding = async (text, signal) => {
  if (embeddingCache.has(text)) return embeddingCache.get(text); // FIX: cache hit

  try {
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "models/text-embedding-004",
          content: { parts: [{ text }] },
        }),
        signal, // FIX: support AbortController
      }
    );
    const data = await response.json();
    if (data.embedding?.values) {
      embeddingCache.set(text, data.embedding.values); // FIX: store in cache
      return data.embedding.values;
    }
    return null;
  } catch (err) {
    if (err.name === "AbortError") throw err;
    console.error("Embedding fetch error:", err);
    return null;
  }
};

// FIX: removed Float32Array conversion inside loop — vecA[i] is already a number
const calculateVectorSimilarity = (vecA, vecB) => {
  if (!vecA || !vecB) return null; // FIX: return null instead of misleading 99.1

  let dotProduct = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (!denom) return null;
  return parseFloat(((dotProduct / denom) * 100).toFixed(1));
};

// ─── Diff Utility (was defined but never used — now wired up) ─────────────────
const diffWords = (original, enhanced) => {
  const oWords = original.trim().split(/\s+/);
  const eWords = enhanced.trim().split(/\s+/);

  // Simple LCS-based diff
  const m = oWords.length, n = eWords.length;
  const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = oWords[i - 1].toLowerCase() === eWords[j - 1].toLowerCase()
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1]);

  const result = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oWords[i - 1].toLowerCase() === eWords[j - 1].toLowerCase()) {
      result.unshift({ type: "same", word: eWords[j - 1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      result.unshift({ type: "added", word: eWords[j - 1] });
      j--;
    } else {
      result.unshift({ type: "removed", word: oWords[i - 1] });
      i--;
    }
  }
  return result;
};

// ─── Reducer ──────────────────────────────────────────────────────────────────
const initialState = {
  output: "",
  changes: [],
  score: null,
  loading: false,
  error: "",
  phase: "idle",
  // FIX: per-tone results cache so switching tone doesn't wipe previous result
  toneResults: {},
};

function reducer(state, action) {
  switch (action.type) {
    case "START":
      return { ...state, loading: true, error: "", phase: "enhancing" };
    case "SCORING":
      return { ...state, phase: "scoring" };
    case "DONE":
      return {
        ...state,
        loading: false,
        phase: "done",
        output: action.output,
        changes: action.changes,
        score: action.score,
        toneResults: {
          ...state.toneResults,
          [action.tone]: { output: action.output, changes: action.changes, score: action.score },
        },
      };
    case "ERROR":
      return { ...state, loading: false, error: action.error, phase: "idle" };
    case "RESET_OUTPUT":
      return { ...state, output: "", changes: [], score: null, phase: "idle", error: "" };
    case "LOAD_TONE_RESULT":
      return {
        ...state,
        output: action.result.output,
        changes: action.result.changes,
        score: action.result.score,
        phase: "done",
      };
    default:
      return state;
  }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ScoreRing({ score }) {
  if (score === null) return null;
  const pct = Math.round(score);
  const r = 38;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  const color = pct >= 90 ? "#4ade80" : pct >= 75 ? "#facc15" : "#f87171";

  return (
    <div className="score-ring-wrap">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={r} fill="none" stroke="#1e293b" strokeWidth="8" />
        <circle
          cx="50" cy="50" r={r} fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          transform="rotate(-90 50 50)"
          style={{ transition: "stroke-dasharray 1s ease" }}
        />
        <text x="50" y="54" textAnchor="middle" fill={color} fontSize="18" fontWeight="700"
          fontFamily="'DM Mono', monospace">{pct}%</text>
      </svg>
      <div className="score-label">Semantic Similarity</div>
    </div>
  );
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

// FIX: DiffView now actually renders the word-level diff
function DiffView({ original, enhanced }) {
  if (!original || !enhanced) return null;
  const tokens = diffWords(original, enhanced);
  return (
    <div className="diff-view">
      {tokens.map((t, i) => (
        <span
          key={i}
          className={`diff-token diff-${t.type}`}
          title={
            t.type === "added" ? "Added"
            : t.type === "removed" ? "Removed"
            : undefined
          }
        >
          {t.word}{" "}
        </span>
      ))}
    </div>
  );
}

// FIX: Skeleton loader for output panel
function Skeleton() {
  return (
    <div className="skeleton-wrap">
      {[80, 100, 65, 90, 50].map((w, i) => (
        <div key={i} className="skeleton-line" style={{ width: `${w}%`, animationDelay: `${i * 0.1}s` }} />
      ))}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [input, setInput] = useState(SAMPLE_TEXT);
  const [toneMode, setToneMode] = useState("professional");
  const [showDiff, setShowDiff] = useState(false);
  const [copied, setCopied] = useState(false);
  const [state, dispatch] = useReducer(reducer, initialState);
  const abortRef = useRef(null); // FIX: AbortController ref

  const selectedTone = TONE_MODES.find((t) => t.id === toneMode);
  const { output, changes, score, loading, error, phase, toneResults } = state;

  // FIX: switching tone loads cached result if available, doesn't wipe previous
  const handleToneSwitch = (id) => {
    setToneMode(id);
    if (toneResults[id]) {
      dispatch({ type: "LOAD_TONE_RESULT", result: toneResults[id] });
    } else {
      dispatch({ type: "RESET_OUTPUT" });
    }
  };

  const handleInputChange = (e) => {
    const val = e.target.value;
    if (val.length > MAX_CHARS) return; // FIX: char limit
    setInput(val);
    dispatch({ type: "RESET_OUTPUT" });
  };

  // FIX: copy to clipboard
  const handleCopy = async () => {
    if (!output) return;
    await navigator.clipboard.writeText(output);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const enhance = useCallback(async () => {
    if (!input.trim()) return;

    // FIX: cancel previous in-flight request
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    dispatch({ type: "START" });

    try {
      const apiKey = import.meta.env.VITE_GROQ_API_KEY;
      const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: "llama-3.3-70b-versatile",
          messages: [
            { role: "system", content: selectedTone.prompt },
            { role: "user", content: `Enhance this text:\n\n${input}` },
          ],
          response_format: { type: "json_object" },
        }),
        signal,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error?.message || "API Error");
      }

      const data = await response.json();
      const raw = data.choices[0].message.content;

      // FIX: safe JSON parse with user-friendly error
      let parsed;
      try {
        const clean = raw.replace(/```json\n?|```/g, "").trim();
        parsed = JSON.parse(clean);
      } catch {
        throw new Error("The AI returned an unexpected format. Please try again.");
      }

      if (!parsed.enhanced) throw new Error("No enhanced text returned. Please try again.");

      dispatch({ type: "SCORING" });

      // FIX: use cached embedding for input if available
      let finalScore = null;
      try {
        const [vecA, vecB] = await Promise.all([
          getEmbedding(input, signal),
          getEmbedding(parsed.enhanced, signal),
        ]);
        finalScore = calculateVectorSimilarity(vecA, vecB); // FIX: returns null on failure, not 99.1
      } catch (embErr) {
        if (embErr.name === "AbortError") throw embErr;
        console.error("Similarity error:", embErr);
        // finalScore stays null — no misleading score shown
      }

      dispatch({
        type: "DONE",
        output: parsed.enhanced,
        changes: parsed.changes || [],
        score: finalScore,
        tone: toneMode,
      });
    } catch (e) {
      if (e.name === "AbortError") return; // silently cancel
      dispatch({ type: "ERROR", error: e.message || String(e) });
    }
  }, [input, selectedTone, toneMode]);

  const isDone = phase === "done" && output;

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
          --yellow: #facc15;
          --text: #e2e8f0;
          --muted: #64748b;
        }

        body {
          background: var(--bg);
          color: var(--text);
          font-family: 'DM Sans', sans-serif;
          min-height: 100vh;
          overflow-x: hidden;
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .app {
          max-width: 960px;
          width: 100%;
          padding: 32px 24px 80px;
        }

        /* ── Header ── */
        .header {
          text-align: center;
          padding: 48px 0 40px;
          display: flex;
          flex-direction: column;
          align-items: center;
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
          font-size: clamp(1.8rem, 4vw, 2.8rem);
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

        /* ── Security notice ── */
        .security-notice {
          display: flex;
          align-items: center;
          gap: 10px;
          background: rgba(250,204,21,0.07);
          border: 1px solid rgba(250,204,21,0.2);
          border-radius: 10px;
          padding: 10px 16px;
          font-size: 12px;
          color: var(--yellow);
          font-family: 'DM Mono', monospace;
          margin-bottom: 24px;
          text-align: left;
        }

        /* ── Tone selector ── */
        .tone-selector {
          display: flex;
          gap: 10px;
          justify-content: center;
          margin-bottom: 28px;
          flex-wrap: wrap;
        }
        .tone-btn {
          font-family: 'DM Sans', sans-serif;
          font-size: 13px;
          font-weight: 500;
          padding: 10px 18px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: var(--surface);
          color: var(--muted);
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 3px;
          position: relative;
        }
        .tone-btn:hover { border-color: var(--accent); color: var(--text); }
        .tone-desc { font-size: 10px; opacity: 0.7; }
        /* FIX: badge showing cached result exists */
        .tone-cached-dot {
          position: absolute;
          top: 6px; right: 6px;
          width: 6px; height: 6px;
          border-radius: 50%;
          background: var(--green);
          box-shadow: 0 0 6px var(--green);
        }

        /* ── Editor grid ── */
        .editor-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          margin-bottom: 24px;
        }
        @media (max-width: 768px) { .editor-grid { grid-template-columns: 1fr; } }

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
          gap: 8px;
          flex-wrap: wrap;
        }
        .panel-title {
          font-family: 'Syne', sans-serif;
          font-size: 13px;
          font-weight: 600;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          color: var(--muted);
        }
        .panel-actions { display: flex; align-items: center; gap: 8px; }
        .panel-badge {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          padding: 2px 10px;
          border-radius: 10px;
          letter-spacing: 0.1em;
        }
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

        .char-bar {
          padding: 8px 20px;
          display: flex;
          justify-content: flex-end;
          align-items: center;
          border-top: 1px solid var(--border);
          gap: 8px;
        }
        .char-count {
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          color: var(--muted);
        }
        .char-count.warn { color: var(--yellow); }
        .char-count.danger { color: var(--red); }
        .char-progress {
          flex: 1;
          height: 2px;
          background: var(--border);
          border-radius: 2px;
          overflow: hidden;
        }
        .char-progress-fill {
          height: 100%;
          border-radius: 2px;
          transition: width 0.2s, background 0.2s;
        }

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
        .placeholder-icon { font-size: 36px; opacity: 0.3; }

        /* ── FIX: Copy button ── */
        .btn-copy {
          font-family: 'DM Sans', sans-serif;
          font-size: 12px;
          font-weight: 500;
          background: rgba(74,222,128,0.1);
          border: 1px solid rgba(74,222,128,0.25);
          color: var(--green);
          padding: 4px 12px;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          gap: 5px;
        }
        .btn-copy:hover { background: rgba(74,222,128,0.2); }

        /* ── FIX: Diff toggle ── */
        .btn-diff {
          font-family: 'DM Sans', sans-serif;
          font-size: 12px;
          font-weight: 500;
          background: rgba(56,189,248,0.08);
          border: 1px solid rgba(56,189,248,0.2);
          color: var(--accent);
          padding: 4px 12px;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
        }
        .btn-diff:hover { background: rgba(56,189,248,0.15); }
        .btn-diff.active {
          background: rgba(56,189,248,0.2);
          border-color: var(--accent);
        }

        /* ── FIX: DiffView ── */
        .diff-view {
          padding: 20px;
          font-size: 15px;
          line-height: 1.9;
          min-height: 260px;
          flex: 1;
        }
        .diff-token { display: inline; }
        .diff-same { color: var(--text); }
        .diff-added {
          background: rgba(74,222,128,0.15);
          color: var(--green);
          border-radius: 3px;
          padding: 1px 3px;
          text-decoration: underline;
          text-decoration-color: rgba(74,222,128,0.4);
        }
        .diff-removed {
          background: rgba(248,113,113,0.12);
          color: var(--red);
          border-radius: 3px;
          padding: 1px 3px;
          text-decoration: line-through;
          text-decoration-color: rgba(248,113,113,0.5);
        }

        /* ── FIX: Skeleton loader ── */
        .skeleton-wrap {
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 12px;
          min-height: 260px;
          justify-content: center;
        }
        .skeleton-line {
          height: 14px;
          border-radius: 6px;
          background: linear-gradient(90deg, var(--surface2) 25%, #1a3050 50%, var(--surface2) 75%);
          background-size: 200% 100%;
          animation: shimmer 1.4s ease infinite;
        }
        @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

        /* ── Action bar ── */
        .action-bar {
          display: flex;
          align-items: center;
          justify-content: center;
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
        .btn-enhance:disabled { opacity: 0.6; cursor: not-allowed; }

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

        .dot-loader { display: flex; gap: 4px; }
        .dot-loader span {
          width: 5px; height: 5px;
          background: var(--accent);
          border-radius: 50%;
          animation: dotBounce 1.2s ease infinite;
        }
        .dot-loader span:nth-child(2) { animation-delay: 0.2s; }
        .dot-loader span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes dotBounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }

        /* ── FIX: Error with retry ── */
        .error-wrap {
          display: flex;
          align-items: center;
          gap: 12px;
          background: rgba(248,113,113,0.08);
          border: 1px solid rgba(248,113,113,0.2);
          border-radius: 10px;
          padding: 12px 16px;
        }
        .error-msg { font-size: 13px; color: var(--red); flex: 1; }
        .btn-retry {
          font-family: 'DM Sans', sans-serif;
          font-size: 12px;
          font-weight: 600;
          background: rgba(248,113,113,0.15);
          border: 1px solid rgba(248,113,113,0.3);
          color: var(--red);
          padding: 6px 14px;
          border-radius: 8px;
          cursor: pointer;
          white-space: nowrap;
          transition: background 0.2s;
        }
        .btn-retry:hover { background: rgba(248,113,113,0.25); }

        /* ── Results ── */
        .results-section { animation: fadeUp 0.5s ease; }
        @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

        .results-header {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 20px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 24px 28px;
          margin-bottom: 20px;
          flex-wrap: wrap;
          text-align: left;
        }
        .score-ring-wrap { text-align: center; flex-shrink: 0; }
        .score-label {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          color: var(--muted);
          letter-spacing: 0.1em;
          text-transform: uppercase;
          margin-top: 4px;
        }
        .score-info { flex: 1; }
        .score-info h3 {
          font-family: 'Syne', sans-serif;
          font-size: 18px;
          font-weight: 700;
          margin-bottom: 6px;
        }
        .score-info p { font-size: 13px; color: var(--muted); line-height: 1.6; }

        .stats-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 14px; }
        .stat-chip {
          background: var(--surface2);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 10px 16px;
          font-family: 'DM Mono', monospace;
        }
        .stat-chip .stat-val { font-size: 20px; font-weight: 500; color: var(--accent); }
        .stat-chip .stat-name { font-size: 10px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }

        /* ── Changelog ── */
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
        .changelog-list { padding: 16px; display: flex; flex-direction: column; gap: 10px; }
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
        .change-number { font-family: 'DM Mono', monospace; font-size: 12px; color: var(--accent); opacity: 0.6; min-width: 24px; padding-top: 2px; }
        .change-content { flex: 1; }
        .change-pair { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
        .change-original {
          font-family: 'DM Mono', monospace; font-size: 12px; color: var(--red);
          background: rgba(248,113,113,0.08); padding: 2px 8px; border-radius: 6px;
          text-decoration: line-through; word-break: break-word;
        }
        .change-enhanced {
          font-family: 'DM Mono', monospace; font-size: 12px; color: var(--green);
          background: rgba(74,222,128,0.08); padding: 2px 8px; border-radius: 6px;
          word-break: break-word;
        }
        .change-arrow { color: var(--muted); font-size: 14px; }
        .change-reason { font-size: 12px; color: var(--muted); line-height: 1.5; }

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

        /* ── Accessibility ── */
        .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
      `}</style>

      <div className="app">

        {/* Header */}
        <div className="header">
          <div className="header-tag">✦ AI Writing Assistant</div>
          <h1>Meaning-Preserving Notes Enhancer</h1>
          <p>Improve your writing without changing what you mean. Grammar, clarity, and structure — never your ideas.</p>
        </div>

        {/* FIX: Security notice reminding devs to proxy API keys in production */}
        <div className="security-notice" role="note">
          ⚠ API keys in env vars are visible client-side. In production, route requests through a backend proxy to keep keys secure.
        </div>

        {/* Tone selector */}
        <div className="tone-selector" role="group" aria-label="Select writing tone">
          {TONE_MODES.map((tone) => (
            <button
              key={tone.id}
              className={`tone-btn ${toneMode === tone.id ? "active" : ""}`}
              style={
                toneMode === tone.id
                  ? { background: tone.color, boxShadow: `0 0 20px ${tone.color}40`, color: "#050a14", fontWeight: 700 }
                  : {}
              }
              onClick={() => handleToneSwitch(tone.id)}
              aria-pressed={toneMode === tone.id}
            >
              {/* FIX: green dot shows a cached result exists for this tone */}
              {toneResults[tone.id] && toneMode !== tone.id && (
                <span className="tone-cached-dot" title="Cached result available" aria-hidden="true" />
              )}
              {tone.icon} {tone.label}
              <span className="tone-desc">{tone.desc}</span>
            </button>
          ))}
        </div>

        {/* Editor grid */}
        <div className="editor-grid">
          {/* Input panel */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Original Text</span>
            </div>
            <textarea
              value={input}
              onChange={handleInputChange}
              placeholder="Paste or type your raw, unpolished text here..."
              aria-label="Original text input"
              maxLength={MAX_CHARS}
            />
            {/* FIX: char count bar */}
            <div className="char-bar">
              <div className="char-progress" aria-hidden="true">
                <div
                  className="char-progress-fill"
                  style={{
                    width: `${(input.length / MAX_CHARS) * 100}%`,
                    background:
                      input.length > MAX_CHARS * 0.9 ? "var(--red)"
                      : input.length > MAX_CHARS * 0.75 ? "var(--yellow)"
                      : "var(--accent)",
                  }}
                />
              </div>
              <span
                className={`char-count ${
                  input.length > MAX_CHARS * 0.9 ? "danger"
                  : input.length > MAX_CHARS * 0.75 ? "warn"
                  : ""
                }`}
                aria-live="polite"
              >
                {input.length}/{MAX_CHARS}
              </span>
            </div>
          </div>

          {/* Output panel */}
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Enhanced Text</span>
              <div className="panel-actions">
                {/* FIX: diff toggle button */}
                {isDone && (
                  <button
                    className={`btn-diff ${showDiff ? "active" : ""}`}
                    onClick={() => setShowDiff((v) => !v)}
                    aria-pressed={showDiff}
                    title="Toggle word-level diff view"
                  >
                    {showDiff ? "← Plain" : "Diff ↔"}
                  </button>
                )}
                {/* FIX: copy button */}
                {isDone && (
                  <button
                    className="btn-copy"
                    onClick={handleCopy}
                    aria-label="Copy enhanced text to clipboard"
                  >
                    {copied ? "✓ Copied!" : "⎘ Copy"}
                  </button>
                )}
                {isDone && <span className="panel-badge badge-enhanced">✓ READY</span>}
              </div>
            </div>

            {/* FIX: skeleton while loading, diff or plain text when done */}
            {loading && phase === "enhancing" ? (
              <Skeleton />
            ) : isDone ? (
              showDiff
                ? <DiffView original={input} enhanced={output} />
                : <div className="output-text">{output}</div>
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

        {/* Action bar */}
        <div className="action-bar">
          <button
            className="btn-enhance"
            onClick={enhance}
            disabled={loading || !input.trim()}
            style={{
              background: `linear-gradient(135deg, ${selectedTone.color}, var(--accent2))`,
              boxShadow: `0 0 24px ${selectedTone.color}50`,
            }}
            aria-label={`Enhance text using ${selectedTone.label} tone`}
          >
            {loading ? <div className="spinner" aria-hidden="true" /> : selectedTone.icon}
            {loading
              ? phase === "enhancing" ? "Enhancing..." : "Scoring..."
              : `Enhance — ${selectedTone.label}`}
          </button>

          {isDone && (
            <button
              className="btn-reset"
              onClick={() => { dispatch({ type: "RESET_OUTPUT" }); }}
            >
              Reset
            </button>
          )}

          {loading && (
            <div className="phase-indicator" aria-live="polite">
              <div className="dot-loader" aria-hidden="true">
                <span /><span /><span />
              </div>
              {phase === "enhancing"
                ? "Analyzing and improving expression..."
                : "Computing semantic similarity..."}
            </div>
          )}

          {/* FIX: error with retry button */}
          {error && (
            <div className="error-wrap" role="alert">
              <span className="error-msg">⚠ {error}</span>
              <button className="btn-retry" onClick={enhance}>Retry ↺</button>
            </div>
          )}
        </div>

        {/* Results */}
        {isDone && (
          <div className="results-section">
            <div className="results-header">
              {/* FIX: score is null when unavailable, not a misleading 99.1 */}
              {score !== null
                ? <ScoreRing score={score} />
                : (
                  <div className="score-ring-wrap" style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 36, opacity: 0.3 }}>⊘</div>
                    <div className="score-label">Score Unavailable</div>
                  </div>
                )
              }

              <div className="score-info">
                <h3>
                  {score === null
                    ? "⚪ Similarity Score Unavailable"
                    : score >= 92 ? "🟢 Meaning Fully Preserved"
                    : score >= 78 ? "🟡 Meaning Largely Preserved"
                    : "🔴 Meaning May Have Shifted"}
                </h3>
                <p>
                  {score === null
                    ? "Semantic scoring failed — check your Gemini API key or quota. The enhanced text was still generated successfully."
                    : score >= 92
                    ? "Strong semantic alignment with your original. Only expression was improved."
                    : score >= 78
                    ? "Close in meaning — review the changes to confirm intent."
                    : "Significant divergence detected. Review the enhanced text closely."}
                </p>
                <div className="stats-row">
                  <div className="stat-chip">
                    <div className="stat-val">{changes.length}</div>
                    <div className="stat-name">Changes</div>
                  </div>
                  {score !== null && (
                    <div className="stat-chip">
                      <div className="stat-val">{score}%</div>
                      <div className="stat-name">Similarity</div>
                    </div>
                  )}
                  <div className="stat-chip">
                    <div className="stat-val">{output.split(/\s+/).length}</div>
                    <div className="stat-name">Words</div>
                  </div>
                  <div className="stat-chip">
                    <div className="stat-val">{toneMode.charAt(0).toUpperCase() + toneMode.slice(1)}</div>
                    <div className="stat-name">Tone</div>
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
