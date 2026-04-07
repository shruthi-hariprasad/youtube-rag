<!-- Copilot / AI agent instructions for the "YouTube Transcript RAG Assistant" project. -->

# Copilot instructions

This file gives targeted, actionable guidance for AI coding agents working on the "YouTube Transcript RAG Assistant" project. Read the project context in `AI_Document_Assistant_Project_Context.md` before making changes.

Core facts (big picture)
- Backend: Python + FastAPI. Expected app entrypoints under a backend package (e.g. `backend/app/main.py` / `backend/app/routes/*.py`).
- Persistence: PostgreSQL stores users, videos and chunk metadata. ChromaDB stores vectors.
- RAG flow: fetch YouTube transcript → chunk (300–400 tokens, ~50 token overlap) → embed (HuggingFace Inference API) → store vectors in ChromaDB (tagged by `video_id`) → retrieve top-K by embedding query → generate with LLM (Anthropic/OpenAI) using chunk context and return source attribution.
- Auth: JWT via `python-jose`. Protected routes: `/videos`, `/query`.

Key files & places to inspect or update
- `AI_Document_Assistant_Project_Context.md` — authoritative project description and development plan.
- Backend entry: look for FastAPI app (commonly `main.py`, `app.py`, or `backend/app/main.py`).
- Database models: search for `users`, `videos`, `chunks` mapping (SQLAlchemy models usually in `models.py` or `db/models/`).
- Vector code: search for `Chroma`, `chromadb`, or `ChromaDB` to find embedding/storage code.

Precise developer workflows & commands (discoverable / expected)
- Local dev (venv):
  - Create env: `python -m venv .venv && source .venv/bin/activate`
  - Install: `pip install -r requirements.txt`
  - Run server (dev): `uvicorn backend.app.main:app --reload --port 8000` (adjust import path as needed).
- Docker: the project plans a `Dockerfile` and `docker-compose.yml` that brings up FastAPI + PostgreSQL + ChromaDB. Preferred quick smoke: `docker-compose up --build` from repo root.
- Tests: look for `pytest` configuration or `tests/`. Run `pytest -q`.

Project-specific conventions and important patterns
- No LangChain: the RAG pipeline is implemented with direct API calls. Expect explicit embedding, indexing and prompt assembly logic rather than an orchestration library.
- Chunking choices matter: transcripts are conversational — use chunk size ~300–400 tokens with ~50 token overlap. Look for chunking utility functions (e.g. `utils/chunking.py`).
- Embeddings: HuggingFace Inference API is used. Look for a single wrapper that takes raw text and returns embeddings; reuse it for both query and chunks.
- Vector storage: Chroma stores vectors; records in PostgreSQL should reference `video_id` and `chunk_index`. Keep metadata in SQL, vectors in Chroma.
- Prompt template should include per-chunk attribution lines (title + channel + snippet) and a short system instruction telling the LLM to cite video titles used.

Integration points & external dependencies
- youtube-transcript-api — fetch transcript text; handle missing captions or private videos.
- YouTube oEmbed (no API key) — fetch title, channel, thumbnail.
- HuggingFace Inference API — embeddings.
- ChromaDB — vector store (likely local docker service in compose).
- LLM: Anthropic or OpenAI — generation.

What an agent should do first (small checklist)
1. Open `AI_Document_Assistant_Project_Context.md` — internalize the planned routes (`/auth/register`, `/auth/login`, `/videos`, `/query`) and data model.
2. Find the FastAPI app entry; run it locally (or run `docker-compose up`) to confirm startup errors.
3. Locate or implement: transcript fetcher, chunker, embedding wrapper, Chroma client, and the retrieval function. Unit-test retrieval with a small transcript fixture.
4. Respect auth boundaries — ensure `user_id` scoping on DB and Chroma queries.

Examples (from project context)
- Chunking example: "Split transcript into overlapping chunks of ~350 tokens with 50-token overlap." Keep this exact sizing when editing related utilities.
- Endpoint examples:
  - `POST /videos` — payload: `{ "url": "https://youtu.be/..." }` — runs teach flow and returns saved video metadata.
  - `POST /query` — payload: `{ "question": "...", "video_ids": [1,2] }` — returns `{ "answer": "...", "sources": [{"video_id":1, "title":"..."}] }`.

Editing & PR guidance for agents
- If adding/changing behavior in the RAG pipeline, include a small unit test that exercises the new behavior using a synthetic transcript (no API calls). Mock external API calls (youtube/HF/LLM/Chroma) in tests.
- Update `.env.example` when introducing new env variables (HUGGINGFACE_KEY, CHROMA_URL, DATABASE_URL, JWT_SECRET, LLM_API_KEY).
- Keep changes small and reviewable — the project goals emphasize explainability and that the author must understand every line.

If you can't find an expected file (e.g., `models.py`, `routes/videos.py`) — ask the repo owner before creating large scaffolding; small, focused additions (utility modules and tests) are fine.

I couldn't find an existing `.github/copilot-instructions.md` or AGENT.md in the repo; this file was created from `AI_Document_Assistant_Project_Context.md`. If you'd like, I can iterate and merge any local agent docs you prefer. What area should I expand next—run/debug commands, test harness, or a starter `docker-compose.yml`? 
