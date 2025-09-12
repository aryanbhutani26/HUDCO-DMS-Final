# HUDCO Document Management System Task


## 🚀 Features (Mapped to Requirements)
- **Content processing & categorization**: Ingest PDFs/Markdown/Text/URLs, chunk & tag resources by topic/type/difficulty.
- **Retrieval with Vector DB**: Uses **Chroma** + **Sentence-Transformers** (default) or **OpenAI embeddings** (optional).
- **Context-aware generation**: Answer questions with retrieved context using **FLAN-T5** (local) .
-
- **UX**: Streamlit app with tabs: Ingest and Chat With PDFs.

## 🧩 Project Structure
```
RAG-Application_Main/
├─ app.py                      # Streamlit UI (demo)
├─ requirements.txt
├─ .env.example                # Put OPENAI_API_KEY here if you want OpenAI
├─ config.yaml                 # Competency graph + domain settings
├─ rag/
│  ├─ ingest.py                # Load, chunk, tag, and persist to Chroma
│  ├─ embeddings.py            # OpenAI or HF Sentence-Transformers
│  ├─ vectorstore.py           # Chroma wrapper
│  ├─ retriever.py             # Top-k semantic retrieval
│  ├─ generator.py             # OpenAI or FLAN-T5 small text generation
├─ data/
│  └─ sample/
│     ├─ content/python_intro.txt
│     ├─ content/python_control_flow.txt
│     ├─ content/python_functions.txt
│     └─ links.json           # Example YouTube/article links by topic
└─ deploy/
   └─ spaces/README.md         # Hugging Face Spaces deploy steps
```

## 🛠️ Setup (Local)
1. **Python** 3.10+ recommended. Create a venv and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. (Optional) **OpenAI**: copy `.env.example` → `.env` and set `OPENAI_API_KEY=...`.
3. **Run the demo**:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL (shown in terminal). Use the **Ingest** tab to index sample content or upload your own.

## ☁️ 1-Click Deploy (Hugging Face Spaces)
- Create a new Space (**Streamlit**).
- Upload the repository files.
- In **Settings → Secrets**, add `OPENAI_API_KEY` if using OpenAI.
- Spaces will auto-install from `requirements.txt` and launch `app.py`.

## 🔧 Configuration
- Edit `config.yaml` to define competencies, prerequisites, and default difficulty per competency.
- Vector store persistence is under `.chroma/` (created at runtime).

## 🧪 Evaluation
- Use **Evaluate** tab to compute `precision@k` and `recall@k` on `evaluation/sample_eval.csv`.
- Add your own annotations (query, relevant_doc_ids list) to grow the eval set.
- Optional RAGAS scaffold included in code comments.

## 📦 Notes
- Default models are light-weight and can run on CPU.
- FLAN-T5 small is used for local generation (downloads at first run). For higher quality, switch to OpenAI in the sidebar.
- This demo is domain-configured for **HUDCO → Python Basics** but you can replace the sample content with your own domain data.


## 🔑 Using Gemini (Google Generative AI)
- Install deps (already in `requirements.txt`): `google-generativeai`
- Copy `.env.example` → `.env` and set `GEMINI_API_KEY=...`
- In the Streamlit sidebar, choose:
  - **Embedding provider** → `gemini` (model default: `text-embedding-004`)
  - **Generator** → `gemini` (model default: `gemini-1.5-flash`)
