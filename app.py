# =========================
# Telemetry hard-off (must be before any Chroma import)
# =========================
import os, logging
os.environ["CHROMA_TELEMETRY__ENABLED"] = "false"  # Chroma global kill
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"       # legacy key some builds check
os.environ["OTEL_SDK_DISABLED"] = "true"           # turn off OpenTelemetry
os.environ["POSTHOG_DISABLED"] = "true"            # extra safety for telemetry libs
os.environ["PH_DISABLED"] = "true"                 # some libs read this too
os.environ.setdefault("GEMINI_CHAT_MODEL", "models/gemini-2.0-flash")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# =========================
# Standard imports
# =========================
import json
import uuid
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
from auth import require_auth, logout
from document_manager import DocumentManager

# =========================
# App bootstrapping
# =========================
load_dotenv()
st.set_page_config(page_title="HUDCO Document Management System", page_icon="🫸", layout="wide")

# Authentication check
if not require_auth():
    st.stop()

CFG_PATH = "config.yaml"
CFG = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

persist_dir = CFG["defaults"]["vector_persist_dir"]
chunk_size = CFG["defaults"]["chunk_size"]
overlap = CFG["defaults"]["chunk_overlap"]
top_k = CFG["defaults"]["top_k"]

provider_embed = "Gemini"
embed_model = "models/text-embedding-004"
provider_gen = "Gemini"

if provider_embed == "Gemini" or provider_gen == "Gemini":
    if not os.getenv("GEMINI_API_KEY"):
        st.sidebar.warning("GEMINI_API_KEY is not set in your environment (.env).")

# =========================
# Core services
# =========================
embedder = Embedder(EmbedConfig(provider=provider_embed, model=embed_model))
vs = VectorStore(persist_dir=persist_dir)
retriever = Retriever(vs, embedder, top_k=top_k)
doc_manager = DocumentManager()

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

    model_name = _normalize_gemini_model(os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.0-flash"))
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
    raw_model = gemini_model_name or os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.0-flash")
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
# UI Header
# =========================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📚 HUDCO Document Management System")
    st.caption(f"Welcome, {st.session_state.username} ({st.session_state.user_role})")

with col2:
    if st.button("🚪 Logout", type="secondary"):
        logout()



# =========================
# Interface Functions
# =========================

def admin_upload_interface():
    """Admin interface for uploading and indexing documents"""
    st.subheader("📤 Upload & Index Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True,
        help="Upload PDF, TXT, or MD files"
    )
    
    description = st.text_area("Description (optional)", placeholder="Brief description of the documents...")
    
    if uploaded_files and st.button("Upload & Index Documents", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save file
            file_id = str(uuid.uuid4())
            file_extension = Path(uploaded_file.name).suffix
            filename = f"{file_id}{file_extension}"
            file_path = doc_manager.upload_dir / filename
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Add to database
            doc_id = doc_manager.add_document(
                filename=filename,
                original_name=uploaded_file.name,
                file_path=str(file_path),
                file_type=file_extension.lstrip('.'),
                file_size=uploaded_file.size,
                uploaded_by=st.session_state.username,
                description=description
            )
            
            # Index the document
            try:
                docs = load_and_chunk([file_path], chunk_size, overlap)
                texts = [d[0] for d in docs]
                metas = [d[1] for d in docs]
                
                # Update metadata with document ID
                for meta in metas:
                    meta['doc_id'] = doc_id
                    meta['original_name'] = uploaded_file.name
                
                ids = [f"doc_{doc_id}_chunk_{j}" for j in range(len(texts))]
                
                embs = embedder.embed(texts)
                vs.add(ids, embs, metas, texts)
                
                # Update index status
                doc_manager.update_index_status(doc_id, True, len(texts))
                
            except Exception as e:
                st.error(f"Error indexing {uploaded_file.name}: {str(e)}")
                doc_manager.update_index_status(doc_id, False, 0)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ All files processed!")
        st.success(f"Successfully uploaded and indexed {len(uploaded_files)} documents!")

def admin_manage_interface():
    """Admin interface for managing documents"""
    st.subheader("📋 Document Management")
    
    documents = doc_manager.get_all_documents()
    
    if not documents:
        st.info("No documents uploaded yet.")
        return
    
    # Display documents in a table
    for doc in documents:
        with st.expander(f"📄 {doc['original_name']} {'✅' if doc['is_indexed'] else '❌'}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Type:** {doc['file_type'].upper()}")
                st.write(f"**Size:** {doc['file_size']:,} bytes")
                st.write(f"**Uploaded by:** {doc['uploaded_by']}")
                st.write(f"**Date:** {doc['upload_date']}")
                if doc['description']:
                    st.write(f"**Description:** {doc['description']}")
                if doc['is_indexed']:
                    st.write(f"**Chunks:** {doc['chunk_count']}")
            
            with col2:
                if not doc['is_indexed']:
                    if st.button(f"🔄 Re-index", key=f"reindex_{doc['id']}"):
                        try:
                            file_path = Path(doc['file_path'])
                            docs = load_and_chunk([file_path], chunk_size, overlap)
                            texts = [d[0] for d in docs]
                            metas = [d[1] for d in docs]
                            
                            for meta in metas:
                                meta['doc_id'] = doc['id']
                                meta['original_name'] = doc['original_name']
                            
                            ids = [f"doc_{doc['id']}_chunk_{j}" for j in range(len(texts))]
                            embs = embedder.embed(texts)
                            vs.add(ids, embs, metas, texts)
                            
                            doc_manager.update_index_status(doc['id'], True, len(texts))
                            st.success("Document re-indexed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{doc['id']}", type="secondary"):
                    if doc_manager.delete_document(doc['id']):
                        st.success("Document deleted!")
                        st.rerun()
                    else:
                        st.error("Failed to delete document")

def user_documents_interface():
    """User interface to view available documents"""
    st.subheader("📄 Available Documents")
    
    documents = doc_manager.get_indexed_documents()
    
    if not documents:
        st.info("No documents are currently available. Please contact an administrator.")
        return
    
    st.write(f"**{len(documents)} documents available for chat:**")
    
    for doc in documents:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 **{doc['original_name']}**")
                if doc['description']:
                    st.caption(doc['description'])
                st.caption(f"Uploaded: {doc['upload_date']} • {doc['chunk_count']} chunks")
            with col2:
                st.write("✅ Ready for chat")
        st.divider()

def user_chat_interface():
    """Chat interface for both admin and users"""
    st.subheader("💬 Chat with Documents")
    
    # Document selection for targeted chat
    documents = doc_manager.get_indexed_documents()
    
    if not documents:
        st.info("No documents available for chat. Upload and index documents first.")
        return
    
    # Document filter
    doc_options = ["All Documents"] + [doc['original_name'] for doc in documents]
    selected_doc = st.selectbox("Select document to chat with:", doc_options)
    
    query = st.text_input("Ask a question about the documents:", placeholder="What would you like to know?")
    
    if st.button("Get Answer", type="primary") and query:
        with st.spinner("Searching and generating answer..."):
            generator = Generator(provider=provider_gen)
            
            # Retrieve relevant chunks
            res = retriever.retrieve(query)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            
            # Filter by selected document if not "All Documents"
            if selected_doc != "All Documents":
                filtered_docs = []
                filtered_metas = []
                for doc, meta in zip(docs, metas):
                    if meta.get('original_name') == selected_doc:
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                docs, metas = filtered_docs, filtered_metas
            
            if not docs:
                st.warning("No relevant information found in the selected document(s).")
                return
            
            # Generate answer
            answer = generator.generate(query, docs, metas)
            
            # Display results
            st.markdown("### 🤖 Answer")
            st.write(answer)
            
            # Show sources
            with st.expander("📚 Sources"):
                unique_sources = list(set(meta.get('original_name', 'Unknown') for meta in metas))
                for source in unique_sources:
                    st.write(f"• {source}")
    
    # Chat history (simple implementation)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if st.session_state.chat_history:
        st.markdown("### 📝 Recent Questions")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-3:])):
            with st.expander(f"Q: {q[:50]}..."):
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {a}")
    
    # Store in history when answer is generated
    if 'answer' in locals() and query:
        st.session_state.chat_history.append((query, answer))


# =========================
# Main UI Logic
# =========================
if st.session_state.user_role == "admin":
    # Admin Interface
    tab_upload, tab_manage, tab_chat = st.tabs(["📤 Upload Documents", "📋 Manage Documents", "💬 Chat"])
    
    with tab_upload:
        admin_upload_interface()
    
    with tab_manage:
        admin_manage_interface()
    
    with tab_chat:
        user_chat_interface()

else:
    # User Interface
    tab_docs, tab_chat = st.tabs(["📄 Available Documents", "💬 Chat with Documents"])
    
    with tab_docs:
        user_documents_interface()
    
    with tab_chat:
        user_chat_interface()