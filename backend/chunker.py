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
            # Carry forward the tail of the current chunk as overlap
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
