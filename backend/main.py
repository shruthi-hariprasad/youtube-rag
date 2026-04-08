from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from urllib.parse import urlparse, parse_qs
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from .database import get_db, engine
from . import models
from .chunker import chunk_text
from .embedder import get_embeddings
from .vector_store import add_chunks


def extract_video_id(url: str) -> str | None:
    """Extract a YouTube video ID from common URL formats.

    Supported formats:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://www.youtube.com/embed/VIDEOID
    - https://www.youtube.com/shorts/VIDEOID

    Returns the video ID string, or None if it can't be determined.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        # short youtu.be links -> path is /VIDEOID
        if hostname.endswith("youtu.be"):
            return parsed.path.lstrip("/") or None

        # full youtube domains
        if "youtube" in hostname:
            # check query string for v=
            qs = parse_qs(parsed.query)
            v = qs.get("v")
            if v:
                return v[0]

            # path-based formats: /embed/VIDEOID, /v/VIDEOID, /shorts/VIDEOID
            parts = [p for p in parsed.path.split("/") if p]
            if not parts:
                return None
            if parts[0] in ("embed", "v", "shorts") and len(parts) >= 2:
                return parts[1]

            # fallback: use the last path segment
            return parts[-1]

    except Exception:
        return None


app = FastAPI()


# Create database tables
models.Base.metadata.create_all(bind=engine)


@app.post("/videos")
def add_video(url: str, db: Session = Depends(get_db)):
    """Add a YouTube video by URL: fetch metadata, transcript, and save to DB.

    Returns the saved Video model instance.
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            # match requested behavior: treat extraction failure as a 400
            raise ValueError("Could not extract video id from URL")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch oEmbed metadata
    oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
    resp = requests.get(oembed_url)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch video metadata from oEmbed")

    try:
        metadata = resp.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON from oEmbed response")

    # Fetch transcript
    try:
        # The youtube-transcript-api package exposes an instance API. Use
        # YouTubeTranscriptApi().fetch(video_id) to retrieve the transcript.
        transcript_list = YouTubeTranscriptApi().fetch(video_id)
        transcript_text = " ".join([t.text for t in transcript_list])
    except Exception as e:
        # Likely no captions or unavailable video
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript: {e}")

    # Persist to DB
    video = models.Video(
        youtube_video_id=video_id,
        title=metadata.get("title"),
        channel_name=metadata.get("author_name"),
        thumbnail_url=metadata.get("thumbnail_url"),
        url=url,
        transcript_text=transcript_text,
    )

    try:
        db.add(video)
        db.commit()
        db.refresh(video)

        # Chunk, embed, and store vectors in the vector store
        chunks = chunk_text(video.transcript_text)
        embeddings = get_embeddings(chunks)
        add_chunks(video.youtube_video_id, chunks, embeddings)

    except Exception as e:
        # Rollback and surface a friendly error
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Could not save video to database: {e}")

    return video


@app.get("/videos")
async def list_videos():
    return {"message": "list endpoint coming soon"}


@app.post("/query")
async def query():
    return {"message": "query endpoint coming soon"}
