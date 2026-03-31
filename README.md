# ✦ Meaning-Preserving Notes Enhancer
## App:->https://my-notes-enhancer.vercel.app
A sophisticated, AI-powered writing assistant that elevates your prose while strictly maintaining your original intent. Built with **React**, **Groq (Llama 3)**, and **Google Gemini Embeddings**.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![Groq](https://img.shields.io/badge/Groq_Cloud-orange?style=flat)
![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=flat&logo=googlegemini&logoColor=white)

---

## 🚀 Overview

Most AI "rewriters" tend to add new information or hallucinate facts. This tool is designed to be a **pure editor**. It uses a dual-engine approach:
1.  **Llama 3.3 (via Groq):** Performs high-speed linguistic enhancement based on specific tone profiles (Casual, Professional, Academic, Email).
2.  **Gemini 1.5 Embeddings:** Calculates a **Semantic Similarity Score** between your original draft and the AI output to ensure your meaning remains $90\%+$ identical.

## ✨ Key Features

* **Four Distinct Tone Modes:** * `Casual Fix`: Grammatical cleanup without losing the "you" in the text.
    * `Professional`: Polished, contraction-free business communication.
    * `Academic`: Scholarly, precise, and passive-voice optimized.
    * `Email`: Direct, polite, and concise.
* **Real-time Change Log:** Every modification is tracked with a "Reason" provided by the AI.
* **Vector Similarity Scoring:** Visual feedback (SVG Gauges) showing how much the core meaning shifted during processing.
* **Modern UI/UX:** A sleek "Dark Mode" interface inspired by high-end developer tools, featuring the **Syne** and **DM Sans** typefaces.

## 🛠️ Tech Stack

* **Frontend:** React (Vite)
* **Styling:** CSS-in-JS (Styled Components approach)
* **LLM (Text Generation):** `llama-3.3-70b-versatile` via **Groq Cloud API**
* **Embeddings (Math/Similarity):** `text-embedding-004` via **Google Gemini API**

---

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/ai-notes-enhancer.git](https://github.com/yourusername/ai-notes-enhancer.git)
    cd ai-notes-enhancer
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    VITE_GROQ_API_KEY=your_groq_api_key_here
    VITE_GEMINI_API_KEY=your_gemini_api_key_here
    ```

4.  **Run the App:**
    ```bash
    npm run dev
    ```

---

## 🧠 How the Similarity Math Works

The app calculates the **Cosine Similarity** between the original text vector ($\vec{A}$) and the enhanced text vector ($\vec{B}$). The result is mapped to a percentage:

$$Similarity = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

* **92% - 100%:** Perfect alignment.
* **80% - 91%:** Minor structural changes, intent is safe.
* **Below 80%:** Significant divergence; user is alerted to review for hallucinations.

## 🤝 Contributing

Contributions are welcome!If you have ideas for new tone prompts or UI improvements, feel free to fork the repo and submit a PR.

---

**Developed with ❤️ by Team ATOM**
