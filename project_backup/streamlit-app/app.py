import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import uuid
import time
from datetime import datetime

from utils.document_processor import extract_text, chunk_text
from utils.embeddings import embed_texts, embed_query
from utils.vector_store import VectorStore
from utils.memory import init_db, save_message, load_history, save_feedback, load_feedback, clear_history
from utils.rag_chain import generate_answer
from utils.evaluator import (
    EVAL_CONTEXT, EVAL_QA, f1_score, partial_match, normalize_text
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind – Intelligent Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content}
if "feedback_map" not in st.session_state:
    st.session_state.feedback_map = {}  # msg_index -> rating
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(persist_path="./chroma_db")
if "collection" not in st.session_state:
    st.session_state.collection = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_running" not in st.session_state:
    st.session_state.eval_running = False

init_db()
session_id = st.session_state.session_id

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 DocuMind")
    st.caption("Intelligent Document Q&A with Memory")
    st.divider()

    # Document status
    st.subheader("📄 Document Status")
    if st.session_state.doc_processed and st.session_state.doc_name:
        st.success(f"**{st.session_state.doc_name}**")
        st.metric("Chunks indexed", len(st.session_state.doc_chunks))
    else:
        st.info("No document loaded")

    st.divider()

    # Conversation history
    st.subheader("🕑 Conversation History")
    history = load_history(session_id, limit=30)
    if history:
        for msg in history:
            role_icon = "🙋" if msg["role"] == "user" else "🤖"
            with st.expander(f"{role_icon} {msg['content'][:60]}…" if len(msg['content']) > 60 else f"{role_icon} {msg['content']}", expanded=False):
                st.write(msg["content"])
                st.caption(msg["timestamp"][:16].replace("T", " "))
        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history(session_id)
            st.session_state.messages = []
            st.session_state.feedback_map = {}
            st.rerun()
    else:
        st.caption("No conversation history yet.")

    st.divider()

    # Feedback summary
    st.subheader("👍 Feedback Summary")
    feedback_records = load_feedback(session_id)
    if feedback_records:
        thumbs_up = sum(1 for f in feedback_records if f["rating"] == 1)
        thumbs_down = sum(1 for f in feedback_records if f["rating"] == -1)
        col1, col2 = st.columns(2)
        col1.metric("👍 Positive", thumbs_up)
        col2.metric("👎 Negative", thumbs_down)
    else:
        st.caption("No feedback yet.")

    st.divider()
    st.caption(f"Session: `{session_id}`")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🧠 Intelligent Document Q&A with Memory")
st.caption("Upload a document, ask questions, and get AI-powered answers using RAG + Gemini")

tab_upload, tab_chat, tab_eval = st.tabs(["📄 Upload & Process", "💬 Chat", "📊 Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Upload & Process
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.header("Upload Your Document")
    st.write("Supported formats: **PDF**, **TXT**, **DOCX**")

    col_up, col_cfg = st.columns([2, 1])

    with col_cfg:
        st.subheader("⚙️ Chunking Settings")
        chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, step=100)
        overlap = st.slider("Overlap (chars)", 0, 400, 150, step=50)
        n_results = st.slider("Top-K retrieval", 1, 8, 4)
        st.session_state.n_results = n_results

    with col_up:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            st.info(f"**{uploaded_file.name}** — {len(file_bytes)/1024:.1f} KB")

            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("Extracting text…"):
                    try:
                        raw_text = extract_text(file_bytes, uploaded_file.name)
                    except Exception as e:
                        st.error(f"Text extraction failed: {e}")
                        st.stop()

                if not raw_text.strip():
                    st.error("No text could be extracted from this document.")
                    st.stop()

                with st.spinner("Chunking text…"):
                    chunks_with_pos = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                    chunk_texts = [c[0] for c in chunks_with_pos]

                st.info(f"Created **{len(chunk_texts)}** chunks. Generating embeddings with Gemini…")

                progress_bar = st.progress(0)
                embeddings = []
                try:
                    batch_size = 5
                    for i in range(0, len(chunk_texts), batch_size):
                        batch = chunk_texts[i : i + batch_size]
                        batch_embeddings = embed_texts(batch)
                        embeddings.extend(batch_embeddings)
                        progress_bar.progress(min((i + batch_size) / len(chunk_texts), 1.0))
                        time.sleep(0.1)
                except Exception as e:
                    st.error(f"Embedding failed: {e}")
                    st.stop()

                progress_bar.empty()

                with st.spinner("Storing in ChromaDB…"):
                    try:
                        collection = st.session_state.vector_store.reset_collection("documents")
                        ids = [f"chunk_{i}" for i in range(len(chunk_texts))]
                        metadatas = [
                            {"source": uploaded_file.name, "chunk_index": i, "char_start": chunks_with_pos[i][1]}
                            for i in range(len(chunk_texts))
                        ]
                        st.session_state.vector_store.add_chunks(
                            collection, chunk_texts, embeddings, metadatas, ids
                        )
                        st.session_state.collection = collection
                    except Exception as e:
                        st.error(f"Vector store error: {e}")
                        st.stop()

                # Save state
                st.session_state.doc_name = uploaded_file.name
                st.session_state.doc_chunks = chunk_texts
                st.session_state.doc_processed = True
                st.session_state.messages = []
                st.session_state.feedback_map = {}

                st.success(f"✅ Document processed! {len(chunk_texts)} chunks indexed into ChromaDB.")
                st.balloons()

    # Show chunk preview
    if st.session_state.doc_processed and st.session_state.doc_chunks:
        st.divider()
        st.subheader("📋 Chunk Preview")
        preview_n = min(5, len(st.session_state.doc_chunks))
        cols = st.columns(min(preview_n, 3))
        for i in range(preview_n):
            with cols[i % 3]:
                with st.expander(f"Chunk {i+1}", expanded=(i == 0)):
                    st.write(st.session_state.doc_chunks[i][:400] + ("…" if len(st.session_state.doc_chunks[i]) > 400 else ""))
        if len(st.session_state.doc_chunks) > preview_n:
            st.caption(f"… and {len(st.session_state.doc_chunks) - preview_n} more chunks")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.doc_processed:
        st.warning("⚠️ Please upload and process a document first (see the **Upload & Process** tab).")
    else:
        st.header(f"Chat about: {st.session_state.doc_name}")

        # Render existing messages
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

                # Show source chunks for assistant messages
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📎 Source chunks used", expanded=False):
                        for j, src in enumerate(msg["sources"]):
                            st.markdown(f"**Chunk {j+1}:** {src[:300]}{'…' if len(src) > 300 else ''}")

                # Feedback buttons for assistant messages
                if msg["role"] == "assistant":
                    current_rating = st.session_state.feedback_map.get(idx)
                    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
                    with fb_col1:
                        up_label = "✅ 👍" if current_rating == 1 else "👍"
                        if st.button(up_label, key=f"up_{idx}", help="This answer was helpful"):
                            if current_rating != 1:
                                st.session_state.feedback_map[idx] = 1
                                q = st.session_state.messages[idx - 1]["content"] if idx > 0 else ""
                                save_feedback(session_id, q, msg["content"], 1)
                                st.rerun()
                    with fb_col2:
                        down_label = "✅ 👎" if current_rating == -1 else "👎"
                        if st.button(down_label, key=f"down_{idx}", help="This answer was not helpful"):
                            if current_rating != -1:
                                st.session_state.feedback_map[idx] = -1
                                q = st.session_state.messages[idx - 1]["content"] if idx > 0 else ""
                                save_feedback(session_id, q, msg["content"], -1)
                                st.rerun()

        # Chat input
        user_input = st.chat_input("Ask a question about your document…")

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            save_message(session_id, "user", user_input)

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    # 1. Embed the query
                    try:
                        q_emb = embed_query(user_input)
                    except Exception as e:
                        st.error(f"Embedding query failed: {e}")
                        st.stop()

                    # 2. Retrieve from ChromaDB
                    try:
                        # Re-open collection if lost (e.g. after rerun)
                        if st.session_state.collection is None:
                            st.session_state.collection = (
                                st.session_state.vector_store.get_or_create_collection("documents")
                            )
                        results = st.session_state.vector_store.query(
                            st.session_state.collection,
                            q_emb,
                            n_results=st.session_state.get("n_results", 4),
                        )
                    except Exception as e:
                        st.error(f"Retrieval failed: {e}")
                        st.stop()

                    top_chunks = results["documents"][0] if results["documents"] else []

                    # 3. Get conversation history
                    hist = load_history(session_id, limit=10)

                    # 4. Generate answer
                    try:
                        answer, _ = generate_answer(user_input, top_chunks, hist)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        st.stop()

                st.write(answer)
                if top_chunks:
                    with st.expander("📎 Source chunks used", expanded=False):
                        for j, src in enumerate(top_chunks):
                            st.markdown(f"**Chunk {j+1}:** {src[:300]}{'…' if len(src) > 300 else ''}")

            # Save assistant message
            assistant_msg = {"role": "assistant", "content": answer, "sources": top_chunks}
            st.session_state.messages.append(assistant_msg)
            save_message(session_id, "assistant", answer)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("📊 RAG Evaluation")
    st.write(
        "Evaluate the Q&A system on a built-in sample dataset drawn from **SQuAD** and **CoQA** style questions. "
        "The evaluation uses a fixed context paragraph about Marie Curie, Python, and the Great Wall of China."
    )

    with st.expander("📄 Evaluation Context", expanded=False):
        st.write(EVAL_CONTEXT)

    with st.expander("📝 Evaluation Questions", expanded=False):
        import pandas as pd
        df_qa = pd.DataFrame([{"Source": q["source"], "Question": q["question"], "Expected": q["expected"]} for q in EVAL_QA])
        st.dataframe(df_qa, width="stretch")

    st.divider()

    if st.button("▶️ Run Evaluation", type="primary", disabled=st.session_state.eval_running):
        st.session_state.eval_running = True
        st.session_state.eval_results = None

        results = []
        vs = st.session_state.vector_store

        # Set up a temporary evaluation collection
        with st.spinner("Setting up evaluation vector store…"):
            try:
                eval_collection = vs.client.get_or_create_collection("eval_collection")
                if eval_collection.count() == 0:
                    eval_chunks_with_pos = chunk_text(EVAL_CONTEXT, chunk_size=300, overlap=50)
                    eval_chunk_texts = [c[0] for c in eval_chunks_with_pos]
                    eval_embeddings = embed_texts(eval_chunk_texts)
                    eval_ids = [f"eval_{i}" for i in range(len(eval_chunk_texts))]
                    eval_metas = [{"source": "eval", "chunk_index": i} for i in range(len(eval_chunk_texts))]
                    vs.add_chunks(eval_collection, eval_chunk_texts, eval_embeddings, eval_metas, eval_ids)
            except Exception as e:
                st.error(f"Eval setup failed: {e}")
                st.session_state.eval_running = False
                st.stop()

        progress = st.progress(0)
        status_text = st.empty()

        for idx, qa in enumerate(EVAL_QA):
            status_text.text(f"Evaluating {idx+1}/{len(EVAL_QA)}: {qa['question'][:60]}…")
            try:
                q_emb = embed_query(qa["question"])
                ret = vs.query(eval_collection, q_emb, n_results=3)
                top_chunks = ret["documents"][0] if ret["documents"] else []
                answer, _ = generate_answer(qa["question"], top_chunks, [])
                em = partial_match(answer, qa["expected"])
                f1 = f1_score(answer, qa["expected"])
                results.append({
                    "Source": qa["source"],
                    "Question": qa["question"],
                    "Expected": qa["expected"],
                    "Predicted": answer[:200] + ("…" if len(answer) > 200 else ""),
                    "Partial Match": "✅" if em else "❌",
                    "F1 Score": round(f1, 3),
                })
            except Exception as e:
                results.append({
                    "Source": qa["source"],
                    "Question": qa["question"],
                    "Expected": qa["expected"],
                    "Predicted": f"Error: {e}",
                    "Partial Match": "❌",
                    "F1 Score": 0.0,
                })
            progress.progress((idx + 1) / len(EVAL_QA))
            time.sleep(0.2)

        status_text.empty()
        progress.empty()
        st.session_state.eval_results = results
        st.session_state.eval_running = False

    # Show results
    if st.session_state.eval_results:
        results = st.session_state.eval_results
        import pandas as pd

        avg_f1 = sum(r["F1 Score"] for r in results) / len(results)
        n_match = sum(1 for r in results if r["Partial Match"] == "✅")

        st.subheader("📈 Results Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Questions", len(results))
        m2.metric("Partial Matches", f"{n_match}/{len(results)}")
        m3.metric("Match Rate", f"{n_match/len(results)*100:.0f}%")
        m4.metric("Avg F1 Score", f"{avg_f1:.3f}")

        # By source
        squad_results = [r for r in results if r["Source"] == "SQuAD"]
        coqa_results = [r for r in results if r["Source"] == "CoQA"]

        col_s, col_c = st.columns(2)
        with col_s:
            st.subheader("SQuAD-style")
            sq_f1 = sum(r["F1 Score"] for r in squad_results) / len(squad_results) if squad_results else 0
            sq_match = sum(1 for r in squad_results if r["Partial Match"] == "✅")
            st.metric("Match Rate", f"{sq_match}/{len(squad_results)}")
            st.metric("Avg F1", f"{sq_f1:.3f}")
        with col_c:
            st.subheader("CoQA-style")
            cq_f1 = sum(r["F1 Score"] for r in coqa_results) / len(coqa_results) if coqa_results else 0
            cq_match = sum(1 for r in coqa_results if r["Partial Match"] == "✅")
            st.metric("Match Rate", f"{cq_match}/{len(coqa_results)}")
            st.metric("Avg F1", f"{cq_f1:.3f}")

        st.divider()
        st.subheader("🔍 Detailed Results")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=400)
