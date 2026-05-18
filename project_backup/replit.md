# DocuMind – Intelligent Document Q&A with Memory

A hackathon-ready RAG-based document Q&A system using Google Gemini, ChromaDB, and SQLite, with a Streamlit frontend.

## Run & Operate

- `cd streamlit-app && streamlit run app.py --server.port 5000` — run the Streamlit app
- Workflow: **Start application** — auto-starts the Streamlit app on port 5000
- Required env: `GEMINI_API_KEY` — Google Gemini API key

## Stack

- Python 3.11 + Streamlit 1.32+
- Google Gemini (`google-genai`): embeddings (`text-embedding-004`) + generation (`gemini-1.5-flash`)
- ChromaDB: local persistent vector database (`./chroma_db`)
- SQLite: conversation memory + feedback storage (`./qa_memory.db`)
- pypdf + python-docx: document text extraction

## Where things live

- `streamlit-app/app.py` — Main Streamlit app (UI, tabs, state management)
- `streamlit-app/utils/document_processor.py` — PDF/TXT/DOCX text extraction + chunking
- `streamlit-app/utils/embeddings.py` — Gemini embedding generation
- `streamlit-app/utils/vector_store.py` — ChromaDB wrapper
- `streamlit-app/utils/memory.py` — SQLite conversation history + feedback
- `streamlit-app/utils/rag_chain.py` — RAG prompt building + Gemini generation
- `streamlit-app/utils/evaluator.py` — SQuAD/CoQA sample evaluation data + metrics

## Architecture decisions

- ChromaDB uses cosine similarity space with pre-computed Gemini embeddings (no built-in embedding fn)
- Conversation history stored in SQLite for persistence across sessions; last 3 Q&A pairs injected into RAG prompt
- Feedback (thumbs up/down) persisted to SQLite per session
- Evaluation uses a fixed built-in context (Marie Curie / Python / Great Wall) so it runs without uploading a doc
- Chunk overlap set to ~150 chars by default to preserve context across boundaries

## Product

- Upload PDF, TXT, or DOCX documents
- Ask questions in a chat interface; answers grounded in the document via RAG
- Conversation history visible in sidebar (short-term memory via SQLite)
- Thumbs up/down feedback on each answer
- Evaluation tab runs SQuAD/CoQA-style benchmark with partial match + F1 metrics

## User preferences

- Python + Streamlit stack
- Simple architecture, working demo over perfection
- Google Gemini for both embeddings and generation

## Gotchas

- `google-genai` (new SDK) is used, not the deprecated `google-generativeai`
- ChromaDB `chroma_db/` and `qa_memory.db` are created at runtime in the working directory
- Streamlit CORS must be disabled (`enableCORS = false`) for Replit proxy to work
