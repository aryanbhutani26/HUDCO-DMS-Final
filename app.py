# =========================
# Telemetry hard-off (must be before any Chroma import)
# =========================
import os, logging
os.environ["CHROMA_TELEMETRY__ENABLED"] = "false"  # Chroma global kill
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"       # legacy key some builds check
os.environ["OTEL_SDK_DISABLED"] = "true"           # turn off OpenTelemetry
os.environ["POSTHOG_DISABLED"] = "true"            # extra safety for telemetry libs
os.environ["PH_DISABLED"] = "true"                 # some libs read this too
os.environ.setdefault("GEMINI_CHAT_MODEL", "models/gemini-1.5-flash")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# =========================
# Standard imports
# =========================
import json
from pathlib import Path
import re

import streamlit as st
import yaml
from dotenv import load_dotenv

from rag.embeddings import Embedder, EmbedConfig
from rag.vectorstore import VectorStore
from rag.ingest import prepare_documents, load_and_chunk
from rag.retriever import Retriever
from rag.generator import Generator
from personalization.learning_styles import QUESTIONS, score, primary_style
from personalization.progress import DB
from personalization.path_planner import plan
from personalization.quiz import generate_quiz, grade_quiz

# =========================
# App bootstrapping
# =========================
load_dotenv()
st.set_page_config(page_title="HUDCO Document Management System", page_icon="📚", layout="wide")

CFG_PATH = "config.yaml"
CFG = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

persist_dir = CFG["defaults"]["vector_persist_dir"]
chunk_size = CFG["defaults"]["chunk_size"]
overlap = CFG["defaults"]["chunk_overlap"]
top_k = CFG["defaults"]["top_k"]
default_style_bias_weight = float(CFG["defaults"].get("style_bias_weight", 0.3))

# =========================
# Sidebar settings (Gemini defaults)
# =========================
st.sidebar.title("⚙️ Settings")

provider_embed = st.sidebar.selectbox(
    "Embedding provider",
    ["Gemini"],
    index=0  # Gemini by default
)

embed_model = st.sidebar.text_input(
    "Embedding model",
    value=(
        "models/text-embedding-004" if provider_embed == "Gemini"
        else (CFG["defaults"]["embedder"] if provider_embed == "sentence-transformers"
              else "text-embedding-3-small")
    ),
    help="For Gemini, use models/text-embedding-004"
)

provider_gen = st.sidebar.selectbox(
    "Generator",
    ["Gemini"],
    index=0  # Gemini by default
)

# style_bias_weight = st.sidebar.slider("Style Bias Weight", 0.0, 1.0, default_style_bias_weight)

if provider_embed == "Gemini" or provider_gen == "Gemini":
    if not os.getenv("GEMINI_API_KEY"):
        st.sidebar.warning("GEMINI_API_KEY is not set in your environment (.env).")

# =========================
# Core services (embedder, vector store, retriever, DB)
# NOTE: Do NOT instantiate Generator here (we lazy-load on click)
# =========================
embedder = Embedder(EmbedConfig(provider=provider_embed, model=embed_model))
vs = VectorStore(persist_dir=persist_dir)
retriever = Retriever(vs, embedder, top_k=top_k)
db = DB(Path("progress.db"))

# =========================
# Helpers
# =========================
def _normalize_gemini_model(name: str) -> str:
    return name if name.startswith(("models/", "tunedModels/")) else f"models/{name}"

def _get_all_docs_from_vs(vs):
    """Fetch all docs/metas from Chroma across minor versions (no 'ids' in include)."""
    # Prefer a VectorStore helper if present
    if hasattr(vs, "get_all") and callable(getattr(vs, "get_all")):
        return vs.get_all()

    col = getattr(vs, "collection", None)
    if col is None:
        return {"documents": [], "metadatas": []}

    try:
        # Chroma 0.5.x supports these keys (NOT 'ids')
        return col.get(include=["documents", "metadatas"])
    except TypeError:
        # Some builds ignore 'include'; just return everything they give
        return col.get()

def generate_quiz_dynamic(comp_id: str, n: int = 4, top_k: int = 4):
    """
    Build a quiz for a competency derived from the uploaded corpus.
    Returns a list of items like:
      {"q":"...", "type":"mcq"|"short", "options":["A","B","C","D"]?, "ans":"..."}
    """
    # Find the competency object created by the Plan step
    comps = st.session_state.get("dyn_competencies", [])
    comp = next((c for c in comps if c.get("id") == comp_id), None)
    if not comp:
        return []

    title = comp.get("title", comp_id)
    objectives = comp.get("objectives", [])
    topic = f"{title}\nObjectives: " + "; ".join(objectives)

    # Retrieve relevant context from the vector store for this competency
    query = f"{title}. {' '.join(objectives)}".strip()
    res = retriever.retrieve(query)
    docs = res.get("documents", [[]])[0]
    context = "\n\n".join(docs)[:8000] if docs else ""

    # Call Gemini to synthesize questions
    import google.generativeai as genai, json, re
    gen_key = os.getenv("GEMINI_API_KEY")
    if not gen_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=gen_key)

    model_name = _normalize_gemini_model(os.getenv("GEMINI_CHAT_MODEL", "models/gemini-1.5-flash"))
    model = genai.GenerativeModel(model_name)

    prompt = f"""
Create {n} short quiz questions for the topic below, based ONLY on the provided context.
For each question return a JSON object with:
- "q": the question text (concise)
- "type": "mcq" or "short"
- "options": a list of 4 options (only if type=="mcq")
- "ans": the correct answer (for mcq, it must be the exact option text; for short, a concise expected answer)

Return STRICT JSON ARRAY ONLY (no markdown, no comments).
If context is too thin, still generate concept-check questions from the topic.

Topic:
{topic}

Context:
{context}
""".strip()

    resp = model.generate_content(prompt)
    text = getattr(resp, "text", str(resp))
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        # fallback: wrap single object if model returned a dict
        m = re.search(r"\{.*\}", text, re.S)
        arr = [json.loads(m.group(0))] if m else []
    else:
        arr = json.loads(m.group(0))

    # minimal normalization
    quiz = []
    for item in arr:
        q = str(item.get("q", "")).strip()
        typ = (item.get("type") or "short").strip().lower()
        options = item.get("options") if typ == "mcq" else None
        ans = str(item.get("ans", "")).strip()
        if q and ans:
            quiz.append({"q": q, "type": ("mcq" if options else "short"), "options": options, "ans": ans})
    return quiz


def build_plan_from_index(
    vs: VectorStore,
    gemini_model_name: str | None = None,
    max_chars_per_source: int = 3500,
    max_chunks_per_source: int = 5,
    min_items: int = 3,
    max_items: int = 12,
):
    """
    Reads all indexed docs from Chroma, groups by source, samples content, and asks Gemini
    to produce a competency list (ids, titles, difficulty, objectives, prerequisites).
    Returns: List[Dict] shaped like config.yaml competencies.
    """
    data = _get_all_docs_from_vs(vs)
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    if not documents:
        return []

    # Group chunks by source file
    by_source = {}
    for d, m in zip(documents, metadatas):
        src = Path(m.get("source", "unknown")).name
        by_source.setdefault(src, []).append(d)

    # Build compact per-source samples
    summaries = []
    for src, chunks in by_source.items():
        sample = "\n".join(chunks[:max_chunks_per_source])[:max_chars_per_source]
        summaries.append({"source": src, "sample": sample})

    # Gemini call
    import google.generativeai as genai
    gem_key = os.getenv("GEMINI_API_KEY")
    if not gem_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=gem_key)
    raw_model = gemini_model_name or os.getenv("GEMINI_CHAT_MODEL", "models/gemini-1.5-flash")
    model_name = _normalize_gemini_model(raw_model)
    model = genai.GenerativeModel(model_name)

    # Prompt
    prompt = f"""
You are building a competency-based learning path from a corpus.

Given the following documents (name + sample text), identify between {min_items} and {max_items} competencies that a learner should master.

Each competency MUST have:
- id: a short slug (kebab-case, unique)
- title: human-friendly title
- difficulty: one of "easy", "medium", "hard"
- objectives: 3–6 bullet points (short sentences) of learning outcomes
- prerequisites: list of other competency ids in this output (empty if none)

Only reference ids that you create in THIS response. Order prerequisites to reflect a logical learning sequence.

Return STRICT JSON ONLY (no markdown, no comments) with this schema:
{{
  "competencies": [
    {{
      "id": "intro-to-x",
      "title": "Intro to X",
      "difficulty": "easy",
      "objectives": ["...","..."],
      "prerequisites": []
    }}
  ]
}}

Documents:
{json.dumps(summaries, ensure_ascii=False)[:120000]}
""".strip()

    resp = model.generate_content(prompt)
    text = getattr(resp, "text", str(resp))
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Gemini did not return JSON. Raw output:\n" + text)

    data_out = json.loads(match.group(0))
    comps = data_out.get("competencies", [])
    # Minimal validation
    for c in comps:
        c.setdefault("prerequisites", [])
        c.setdefault("objectives", [])
        c.setdefault("difficulty", "medium")
    return comps

# =========================
# UI Tabs
# =========================
st.title("HUDCO Document Management System  Demo (Gemini)")

tab_ingest, tab_learn = st.tabs(
    ["Ingest", "Chat With Ingested Files"]
)

# ---------------------------
# Ingest Tab
# ---------------------------
with tab_ingest:
    st.subheader("Ingest Your Files")
    st.write("Upload `.pdf`, `.txt`, or `.md` files. Or index the included sample content.")

    uploaded = st.file_uploader("Upload files", type=["pdf", "txt", "md"], accept_multiple_files=True)

    if st.button("Index Uploaded Files") and uploaded:
        paths = []
        temp_dir = Path("uploaded")
        temp_dir.mkdir(exist_ok=True)
        for f in uploaded:
            p = temp_dir / f.name
            p.write_bytes(f.read())
            paths.append(p)

        docs = load_and_chunk(paths, chunk_size, overlap)
        texts = [d[0] for d in docs]
        metas = [d[1] for d in docs]
        ids = [f"{m['source']}::{i}" for i, m in enumerate(metas)]

        embs = embedder.embed(texts)
        vs.add(ids, embs, metas, texts)
        st.success(f"Ingested {len(texts)} chunks from {len(paths)} files.")

    if st.button("Index Sample Content"):
        content_dir = Path("data/sample/content")
        paths = prepare_documents(content_dir)
        docs = load_and_chunk(paths, chunk_size, overlap)
        texts = [d[0] for d in docs]
        metas = [d[1] for d in docs]

        def nice_id(m, i):
            src = Path(m['source']).name.replace(".pdf", "").replace(".txt", "").replace(".md", "")
            return f"{src}.txt::{i}"

        ids = [nice_id(m, i) for i, m in enumerate(metas)]
        embs = embedder.embed(texts)
        vs.add(ids, embs, metas, texts)
        st.success(f"Ingested {len(texts)} chunks from sample content.")

    if st.button("Reset Vector Store"):
        vs.reset()
        st.warning("Vector store cleared.")
# ---------------------------
# Learn / Chat Tab
# ---------------------------
with tab_learn:
    st.subheader("Learn / Chat with RAG")
    name4 = st.text_input("Name", value="HUDCO", key="chat_name")
    user = db.get_user(name4)
    query = st.text_input("Ask a question")

    if st.button("Retrieve & Answer") and query:
        # Lazy-load the generator at click time so no HF models initialize early
        generator = Generator(provider=provider_gen)

        # Retrieve
        res = retriever.retrieve(query)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]

        # Simple style-aware re-ranking (demo only)
        sty = user["style"] if user else "Unknown"

        def style_score(meta):
            src = meta.get("source", "")
            if sty == "Visual" and ("png" in src or "jpg" in src or "mp4" in src or "video" in src):
                return 1
            if sty == "Auditory" and ("mp3" in src or "podcast" in src or "video" in src):
                return 1
            if sty == "Read/Write" and (src.endswith(".txt") or src.endswith(".md") or src.endswith(".pdf")):
                return 1
            if sty == "Kinesthetic":
                return 0.2
            return 0

        if metas:
            pairs = list(zip(docs, metas, ids))
            # Light re-rank by style (you can scale by style_bias_weight if you like)
            pairs.sort(key=lambda x: style_score(x[1]), reverse=True)
            docs, metas, ids = zip(*pairs)

        # Generate answer
        answer = generator.generate(query, list(docs), list(metas))
        st.markdown("**Answer**")
        st.write(answer)
        with st.expander("Sources"):
            for m in metas:
                st.write(m.get("source"))
        
    if st.button("Open Gemini in Browser"):
        st.markdown(
            """
            <meta http-equiv="refresh" content="0; url='https://gemini.google.com/'">
            """,
            unsafe_allow_html=True
        )

