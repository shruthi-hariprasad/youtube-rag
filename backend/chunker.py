import re
from typing import List


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def chunk_text(text: str) -> List[str]:
    if not text:
        return []

    sentences = _split_sentences(text)
    chunk_size = 300
    overlap_size = 50
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        sw = len(sentence.split())
        if current_words + sw > chunk_size and current:
            chunks.append(" ".join(current))
            overlap: List[str] = []
            ow = 0
            for s in reversed(current):
                w = len(s.split())
                if ow + w <= overlap_size:
                    overlap.insert(0, s)
                    ow += w
                else:
                    break
            current = overlap
            current_words = ow
        current.append(sentence)
        current_words += sw

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_segments(segments: list[dict]) -> list[tuple[str, float]]:
    """Split transcript segments into overlapping chunks, preserving start timestamps.

    Returns list of (chunk_text, start_time_seconds) tuples where start_time is
    the timestamp of the first segment in each chunk.
    """
    if not segments:
        return []

    chunk_size = 300
    overlap_size = 50
    results: list[tuple[str, float]] = []
    current: list[dict] = []
    current_words = 0

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        sw = len(text.split())

        if current_words + sw > chunk_size and current:
            chunk_str = " ".join(s["text"] for s in current)
            chunk_start = float(current[0]["start"])
            results.append((chunk_str, chunk_start))

            overlap: list[dict] = []
            ow = 0
            for s in reversed(current):
                w = len(s["text"].split())
                if ow + w <= overlap_size:
                    overlap.insert(0, s)
                    ow += w
                else:
                    break
            current = overlap
            current_words = ow

        current.append({"text": text, "start": float(seg.get("start", 0.0))})
        current_words += sw

    if current:
        results.append((" ".join(s["text"] for s in current), float(current[0]["start"])))

    return results
