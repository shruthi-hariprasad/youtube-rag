"""Simple chunking utility for splitting transcript text into overlapping chunks.

The public function `chunk_text` follows the project convention:
- chunk size: 300 words
- overlap: 50 words

It splits on whitespace using `str.split()` and returns a list of chunk strings.
"""
from typing import List


def chunk_text(text: str) -> List[str]:
    """Split `text` into overlapping chunks of words.

    Args:
        text: Input string (transcript or document).

    Returns:
        A list of strings where each string is a chunk of up to `chunk_size`
        words. Chunks overlap by `overlap` words.
    """
    if not text:
        return []

    words = text.split()
    chunk_size = 300
    overlap = 50
    results: List[str] = []

    # Prevent negative or zero step in pathological configs
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    start = 0
    n = len(words)
    while start < n:
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        results.append(chunk)
        start += step

    return results
