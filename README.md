# YouTube Transcript RAG Assistant

A full-stack AI-powered web application that lets you build a personal YouTube knowledge base. Paste any YouTube URL, and the app automatically fetches the transcript. Ask questions across all your saved videos and get grounded answers that tell you exactly which video they came from.

**Status: In progress - backend complete, frontend and deployment in progress**

---

## Demo

*Live demo and GIF coming soon*

---

## What It Does

- **Paste a YouTube URL** → transcript is fetched automatically via `youtube-transcript-api`
- **Cross-video search** → ask a question and the app searches across your entire video library
- **Grounded answers** → the LLM only answers from transcript content, with source attribution
- **Personal library** → JWT-based auth means your video library is private to you

---

## How It Works

```
YouTube URL
    ↓
Transcript fetch (youtube-transcript-api)
    ↓
Text chunking with overlap (300 words, 50 word overlap)
    ↓
Embedding (HuggingFace all-MiniLM-L6-v2)
    ↓
Vector storage (ChromaDB)

At query time:
Question → embed → ChromaDB semantic search → top 5 chunks → LLM (Groq/Llama 3) → answer + sources
```

---

## Tech Stack

**Backend**
- Python, FastAPI, SQLAlchemy, PostgreSQL
- youtube-transcript-api (automatic transcript fetching)
- ChromaDB (vector storage, cosine similarity)
- HuggingFace Inference API (sentence-transformers/all-MiniLM-L6-v2)
- Groq API / Llama 3 (LLM generation)
- JWT auth with python-jose and bcrypt

**Frontend** *(in progress)*
- React, TypeScript, Vite, Tailwind CSS, Axios

**Infrastructure** *(in progress)*
- Docker + docker-compose
- Render (backend), Vercel (frontend)

---

## Run Locally

### Prerequisites
- Python 3.12+
- PostgreSQL
- Node.js 18+

### Backend

```bash
# Clone the repo
git clone https://github.com/shruthi-hariprasad/youtube-rag.git
cd youtube-rag/backend

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in DATABASE_URL, SECRET_KEY, HF_TOKEN, GROQ_API_KEY

# Create database
createdb youtube_rag

# Run server
cd ..
uvicorn backend.main:app --reload
```

API docs available at `http://localhost:8000/docs`

### Frontend *(once complete)*

```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/auth/register` | Create account | No |
| POST | `/auth/login` | Login, returns JWT | No |
| POST | `/videos` | Add video by YouTube URL | Yes |
| GET | `/videos` | List your video library | Yes |
| POST | `/query` | Ask a question | Yes |

---

## Project Structure

```
youtube-rag/
├── backend/
│   ├── main.py          # FastAPI app and endpoints
│   ├── database.py      # SQLAlchemy setup
│   ├── models.py        # User and Video models
│   ├── auth.py          # JWT and password hashing
│   ├── chunker.py       # Text chunking with overlap
│   ├── embedder.py      # HuggingFace embeddings
│   ├── vector_store.py  # ChromaDB client
│   ├── retriever.py     # Semantic search
│   ├── generator.py     # LLM generation
│   └── tests/           # pytest test suite (8 tests)
└── frontend/            # React TypeScript (in progress)
```

---

## Known Limitations

- Videos without auto-generated captions (rare for popular creators) cannot be transcribed
- Free API tier rate limits may slow down embedding for very long videos
- Retrieval quality drops when query vocabulary is very different from transcript vocabulary (a known limitation of dense retrieval)

---

*Built by [Shruthi Hariprasad](https://github.com/shruthi-hariprasad) — MS CS, UMass Amherst*
