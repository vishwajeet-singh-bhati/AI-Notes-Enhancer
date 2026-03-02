# ✦ Meaning-Preserving AI Notes Enhancer

> GDG Hackathon Solasta 2026 | IIITDM Kurnool  
> Problem Statement 3 — Meaning-Preserving AI Notes Enhancer

A web-based AI writing assistant that improves grammar, spelling, clarity, and sentence structure — **without ever changing your meaning.**

---

## 🚀 Live Demo
> Add your Vercel deployment URL here after deploying

---

## 🧠 Agent Details

| Field | Value |
|---|---|
| **Agent Name** | Meaning-Preserving Notes Enhancer |
| **Framework** | React 18 + Vite |
| **AI Model** | Claude Haiku (claude-haiku-4-5) via Anthropic API |
| **Similarity Method** | TF-based Cosine Similarity (client-side) |

---

## ⚙️ Architecture

```
User Input
    ↓
React Frontend (Vite)
    ↓
Anthropic API (Claude Haiku)
    ↓ (JSON response)
├── Enhanced Text
├── Change Log (original → improved + reason)
└── Semantic Similarity Score (cosine similarity on TF vectors)
    ↓
Side-by-side comparison view + Score Ring + Change Log UI
```

**Key design decisions:**
- The system prompt is engineered to enforce meaning preservation as a hard constraint
- Auto-retry logic (up to 3x) handles API overload gracefully  
- Cosine similarity is computed client-side on TF vectors — no extra API calls needed
- All changes are explained with rationale in the transparent change log

---

## 🛠️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/notes-enhancer.git
cd notes-enhancer
```

### 2. Install dependencies
```bash
npm install
```

### 3. Add your Anthropic API key
```bash
cp .env.example .env
```
Then edit `.env` and replace `your_api_key_here` with your actual key from [console.anthropic.com](https://console.anthropic.com).

### 4. Run locally
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173)

---

## ▲ Deploy to Vercel

### Option A — Vercel CLI (fastest)
```bash
npm install -g vercel
vercel
```
When prompted, add your environment variable:
- **Key:** `VITE_ANTHROPIC_API_KEY`
- **Value:** your Anthropic API key

### Option B — GitHub + Vercel Dashboard
1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **New Project** → Import your repo
3. In **Environment Variables**, add:
   - `VITE_ANTHROPIC_API_KEY` = your key
4. Click **Deploy** ✓

---

## ✅ Features

- **Side-by-side editor** — original text left, enhanced text right
- **Semantic similarity score** — visual ring gauge with % and color verdict
- **Transparent change log** — every modification with before/after and rationale
- **Auto-retry** — handles API overload automatically (up to 3 retries)
- **Meaning preservation** — strict system prompt prevents idea addition/removal
- **Professional tone elevation** — expands contractions, removes colloquialisms

---

## 📊 Evaluation Criteria Coverage

| Criterion | Implementation |
|---|---|
| Meaning preservation (30%) | Hard-constrained system prompt |
| Grammar & fluency (25%) | Claude Haiku with detailed rules |
| Similarity validation (20%) | TF cosine similarity with visual ring |
| Change log transparency (15%) | Per-change rationale displayed |
| UI/UX (10%) | Polished dark UI with animations |
