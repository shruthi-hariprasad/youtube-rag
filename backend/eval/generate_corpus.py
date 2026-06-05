"""
One-time script: pulls transcripts from DB, generates QA pairs via Groq,
saves backend/eval/corpus.json.

Usage:
    python -m backend.eval.generate_corpus

Outputs corpus.json with ~75 QA pairs across 4 video types.
Check corpus.json into git so CI never needs DB access.
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sqlalchemy.orm import Session

load_dotenv(Path(__file__).parent.parent / ".env")

CORPUS_PATH = Path(__file__).parent / "corpus.json"
VIDEO_IDS = ["MjnFMbhn7f0", "mbd2NdZ4Cl4", "o1at94k4MXM", "Z2R7bspWSPo"]
QA_PER_VIDEO = 19  # ~76 total

_SYSTEM = textwrap.dedent("""
    You are a QA corpus generator for a RAG evaluation harness.
    Given a transcript excerpt, generate factual question-answer pairs
    that test whether a RAG system can answer from the transcript.

    Rules:
    - Each question must be answerable ONLY from the provided text
    - Answers must be specific (1-3 sentences), not vague
    - Vary question types: factual, "how does X work", "what happened when", "who said"
    - Do NOT generate questions about things not mentioned in the text
    - Return a JSON array only, no other text:
      [{"question": "...", "ground_truth": "..."}, ...]
""").strip()


def _generate_qa(client: Groq, title: str, transcript_chunk: str, n: int) -> list[dict]:
    prompt = (
        f"Video title: {title}\n\n"
        f"Transcript excerpt:\n{transcript_chunk}\n\n"
        f"Generate exactly {n} diverse question-answer pairs from this transcript."
    )
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()
    return json.loads(raw)


def generate(db: Session) -> list[dict]:
    from backend.models import Video

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    corpus: list[dict] = []

    for vid_id in VIDEO_IDS:
        video = db.query(Video).filter(Video.youtube_video_id == vid_id).first()
        if not video or not video.transcript_text:
            print(f"  SKIP {vid_id}: no transcript")
            continue

        title = video.title or vid_id
        # Use middle 6000 words for richer content (avoids intro/outro noise)
        words = video.transcript_text.split()
        start = max(0, len(words) // 4)
        excerpt = " ".join(words[start : start + 6000])

        print(f"  Generating {QA_PER_VIDEO} QA pairs for: {title[:60]}")
        try:
            pairs = _generate_qa(client, title, excerpt, QA_PER_VIDEO)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        for pair in pairs:
            pair["video_id"] = vid_id
            pair["video_title"] = title
        corpus.extend(pairs)
        print(f"    -> {len(pairs)} pairs generated")

    return corpus


if __name__ == "__main__":
    from backend.database import SessionLocal

    print("Generating eval corpus...")
    db = SessionLocal()
    try:
        corpus = generate(db)
    finally:
        db.close()

    CORPUS_PATH.write_text(json.dumps(corpus, indent=2))
    print(f"\nSaved {len(corpus)} QA pairs to {CORPUS_PATH}")
