from contextlib import asynccontextmanager
import asyncio
import json
import logging
import re
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from urllib.parse import urlparse, parse_qs
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from .database import get_db, engine, SessionLocal
from . import models
from .chunker import chunk_text, chunk_segments
from .embedder import get_embeddings
from .vector_store import add_chunks, delete_chunks, get_collection
from .retriever import retrieve_chunks
from .generator import generate_answer, stream_answer, generate_summary_and_questions
from .agent import run_agent
from pydantic import BaseModel
from .auth import hash_password, verify_password, create_token, decode_token
import os

logger = logging.getLogger(__name__)


class UserCreate(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class QueryRequest(BaseModel):
    question: str
    video_id: str | None = None
    history: list[dict] | None = None




security_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme)
) -> int:
    user_id = decode_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id


def extract_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        if hostname.endswith("youtu.be"):
            return parsed.path.lstrip("/") or None

        if "youtube" in hostname:
            qs = parse_qs(parsed.query)
            v = qs.get("v")
            if v:
                return v[0]

            parts = [p for p in parsed.path.split("/") if p]
            if not parts:
                return None
            if parts[0] in ("embed", "v", "shorts") and len(parts) >= 2:
                return parts[1]

            return parts[-1]

    except Exception:
        return None


def _reembed_if_empty():
    if get_collection().count() > 0:
        return
    db = SessionLocal()
    try:
        videos = db.query(models.Video).all()
        for video in videos:
            if video.transcript_segments:
                segs = json.loads(video.transcript_segments)
                pairs = chunk_segments(segs)
                chunks = [p[0] for p in pairs]
                times = [p[1] for p in pairs]
            elif video.transcript_text:
                chunks = chunk_text(video.transcript_text)
                times = None
            else:
                continue
            embeddings = get_embeddings(chunks)
            add_chunks(video.youtube_video_id, chunks, embeddings, start_times=times)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_reembed_if_empty)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://youtube-rag-mu.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)


@app.post("/auth/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if not re.match(r'^[a-zA-Z0-9_.-]{3,30}$', user.username):
        raise HTTPException(status_code=400, detail="Username must be 3–30 characters: letters, numbers, _ . -")
    existing = db.query(models.User).filter(models.User.email == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    new_user = models.User(
        email=user.username,
        hashed_password=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}


@app.post("/auth/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(db_user.id)
    return {"access_token": token, "token_type": "bearer"}


@app.put("/auth/password")
def change_password(body: PasswordChange, db: Session = Depends(get_db), user_id: int = Depends(get_current_user)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not verify_password(body.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    user.hashed_password = hash_password(body.new_password)
    db.commit()
    return {"message": "Password updated"}


@app.delete("/auth/account")
def delete_account(db: Session = Depends(get_db), user_id: int = Depends(get_current_user)):
    videos = db.query(models.Video).filter(models.Video.user_id == user_id).all()
    for video in videos:
        delete_chunks(video.youtube_video_id)
    db.query(models.Video).filter(models.Video.user_id == user_id).delete()
    db.query(models.User).filter(models.User.id == user_id).delete()
    db.commit()
    return {"message": "Account deleted"}


def _ingest_video(video_db_id: int) -> None:
    """Embed chunks and generate summary/questions in the background."""
    db = SessionLocal()
    try:
        video = db.get(models.Video, video_db_id)
        if not video:
            return

        pairs = chunk_segments(json.loads(video.transcript_segments))
        chunks = [p[0] for p in pairs]
        times = [p[1] for p in pairs]
        embeddings = get_embeddings(chunks)
        add_chunks(video.youtube_video_id, chunks, embeddings, start_times=times)

        try:
            sq = generate_summary_and_questions(video.transcript_text)
            video.summary = sq.get("summary", "")
            video.suggested_questions = json.dumps(sq.get("questions", []))
            db.commit()
        except Exception:
            logger.exception("Failed to generate summary/questions for video %s", video_db_id)
    except Exception:
        logger.exception("Background ingestion failed for video %s", video_db_id)
    finally:
        db.close()


@app.post("/videos", status_code=202)
def add_video(
    url: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user),
):
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Could not extract video id from URL")

    existing = db.query(models.Video).filter(
        models.Video.youtube_video_id == video_id,
        models.Video.user_id == user_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="You have already added this video")

    oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
    resp = requests.get(oembed_url)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch video metadata from oEmbed")

    try:
        metadata = resp.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON from oEmbed response")

    try:
        proxy_username = os.getenv("WEBSHARE_PROXY_USERNAME")
        proxy_password = os.getenv("WEBSHARE_PROXY_PASSWORD")

        def _fetch(use_proxy: bool):
            if use_proxy and proxy_username and proxy_password:
                from youtube_transcript_api.proxies import WebshareProxyConfig
                ytt = YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(
                        proxy_username=proxy_username,
                        proxy_password=proxy_password,
                    )
                )
            else:
                ytt = YouTubeTranscriptApi()
            return ytt.fetch(video_id)

        try:
            transcript_list = _fetch(use_proxy=True)
        except Exception:
            transcript_list = _fetch(use_proxy=False)

        segments = [{"text": t.text, "start": t.start} for t in transcript_list]
        transcript_text = " ".join(s["text"] for s in segments)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript: {e}")

    video = models.Video(
        user_id=user_id,
        youtube_video_id=video_id,
        title=metadata.get("title"),
        channel_name=metadata.get("author_name"),
        thumbnail_url=metadata.get("thumbnail_url"),
        url=url,
        transcript_text=transcript_text,
        transcript_segments=json.dumps(segments),
    )

    try:
        db.add(video)
        db.commit()
        db.refresh(video)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Could not save video to database: {e}")

    background_tasks.add_task(_ingest_video, video.id)
    return video


@app.get("/videos")
def get_videos(db: Session = Depends(get_db), user_id: int = Depends(get_current_user)):
    videos = db.query(models.Video).filter(models.Video.user_id == user_id).all()
    return videos


@app.delete("/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db), user_id: int = Depends(get_current_user)):
    video = db.query(models.Video).filter(
        models.Video.id == video_id,
        models.Video.user_id == user_id
    ).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    delete_chunks(video.youtube_video_id)
    db.delete(video)
    db.commit()
    return {"message": "Video deleted successfully"}


def _get_enriched_chunks(req: QueryRequest, db: Session, user_id: int) -> list[dict]:
    user_videos = db.query(models.Video).filter(models.Video.user_id == user_id).all()
    user_video_ids = [v.youtube_video_id for v in user_videos]
    filter_ids = [req.video_id] if req.video_id else user_video_ids

    if not filter_ids:
        raise HTTPException(status_code=404, detail="No videos in your library yet")

    chunks = retrieve_chunks(req.question, video_ids=filter_ids)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found")

    unique_video_ids = list(set(c["video_id"] for c in chunks))
    videos = db.query(models.Video).filter(
        models.Video.youtube_video_id.in_(unique_video_ids)
    ).all()
    title_map = {v.youtube_video_id: v.title for v in videos}

    for chunk in chunks:
        chunk["title"] = title_map.get(chunk["video_id"], chunk["video_id"])

    return chunks


@app.post("/query")
def query(req: QueryRequest, db: Session = Depends(get_db), user_id: int = Depends(get_current_user)):
    chunks = _get_enriched_chunks(req, db, user_id)
    return generate_answer(req.question, chunks, history=req.history)


@app.post("/query/stream")
def query_stream(req: QueryRequest, db: Session = Depends(get_db), user_id: int = Depends(get_current_user)):
    chunks = _get_enriched_chunks(req, db, user_id)
    return StreamingResponse(
        stream_answer(req.question, chunks, history=req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/agent")
def query_agent(req: QueryRequest, db: Session = Depends(get_db), user_id: int = Depends(get_current_user)):
    user_videos = db.query(models.Video).filter(models.Video.user_id == user_id).all()
    user_video_ids = [v.youtube_video_id for v in user_videos]
    filter_ids = [req.video_id] if req.video_id else user_video_ids
    title_map = {
        v.youtube_video_id: f"{v.title} (channel: {v.channel_name})" if v.channel_name else v.title
        for v in user_videos
    }
    # Metadata chunks let the synthesizer answer questions about the video itself
    # (title, channel) that would never appear in the transcript text
    meta_chunks = [
        {
            "video_id": v.youtube_video_id,
            "title": title_map[v.youtube_video_id],
            "text": f"Video title: {v.title}\nChannel: {v.channel_name or 'unknown'}\nURL: {v.url}"
                    + (f"\nSummary: {v.summary}" if v.summary else ""),
            "source": "video",
            "chunk_index": -1,
            "start_time": 0.0,
        }
        for v in user_videos
        if v.youtube_video_id in filter_ids
    ]

    if not filter_ids:
        raise HTTPException(status_code=404, detail="No videos in your library yet")

    return StreamingResponse(
        run_agent(req.question, filter_ids, title_map, meta_chunks, history=req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
