"""
Retrieval evaluation harness.

Measures whether the correct chunk appears in top-K retrieval results
using hand-labelled question-answer pairs from a sample transcript.

Usage:
    # BM25-only, no API keys needed:
    python -m backend.eval.eval_harness --demo

    # Full semantic + hybrid evaluation (requires HF_TOKEN + GROQ_API_KEY):
    python -m backend.eval.eval_harness --full

    # Evaluate against a live video in your DB:
    python -m backend.eval.eval_harness --video-id <youtube_video_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

_SAMPLE_PATH = Path(__file__).parent / "sample_transcript.json"

# Hand-labelled QA pairs: (question, ground_truth_chunk_index)
# 4 chunks produced by the 300-word chunker with 50-word overlap:
# chunk 0: gradient descent definition, loss function, learning rate intro + "too large" warning
# chunk 1: learning rate range test, cosine annealing, batch/SGD/mini-batch variants, backprop intro
# chunk 2: backprop chain rule, forward/backward pass, vanishing gradient, Adam definition
# chunk 3: Adam hyperparameters, RMSprop, L1/L2 regularization, dropout
QA_PAIRS: list[tuple[str, int]] = [
    ("What is gradient descent?", 0),
    ("What does the loss function measure?", 0),
    ("What happens if the learning rate is too large?", 0),
    ("How does the learning rate range test work?", 1),
    ("What is the difference between batch and stochastic gradient descent?", 1),
    ("What batch size is typically used for mini-batch gradient descent?", 1),
    ("How does backpropagation use the chain rule?", 2),
    ("What is the vanishing gradient problem?", 2),
    ("How does the Adam optimizer work?", 2),
    ("What is the difference between L1 and L2 regularization?", 3),
    ("How does dropout prevent overfitting?", 3),
]


def _chunk_segments(segments: list[dict]) -> list[tuple[str, float]]:
    """Mirrors backend/chunker.py chunk_segments without importing it."""
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
            results.append((" ".join(s["text"] for s in current), float(current[0]["start"])))
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


def _bm25_retrieve(query: str, chunks: list[str], n: int = 5) -> list[int]:
    """Return indices of top-n BM25 results."""
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]


def _semantic_retrieve(query: str, chunks: list[str], n: int = 5) -> list[int]:
    """Return indices of top-n semantic results via HuggingFace API."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from backend.embedder import get_embeddings
    import numpy as np

    all_embeddings = get_embeddings(chunks + [query])
    query_emb = np.array(all_embeddings[-1])
    chunk_embs = np.array(all_embeddings[:-1])
    norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    chunk_embs = chunk_embs / np.maximum(norms, 1e-8)
    query_norm = np.linalg.norm(query_emb)
    query_emb = query_emb / max(query_norm, 1e-8)
    scores = chunk_embs @ query_emb
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]


def _rrf(bm25_ranking: list[int], sem_ranking: list[int], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over two ranked lists of chunk indices."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(sem_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def _hit_at_k(results: list[int], ground_truth: int, k: int) -> bool:
    return ground_truth in results[:k]


def _reciprocal_rank(results: list[int], ground_truth: int) -> float:
    for rank, idx in enumerate(results, start=1):
        if idx == ground_truth:
            return 1.0 / rank
    return 0.0


def _print_table(title: str, rows: list[tuple[str, float]]) -> None:
    col_w = max(len(r[0]) for r in rows) + 2
    print(f"\n{title}")
    print("-" * (col_w + 10))
    for label, val in rows:
        print(f"  {label:<{col_w}} {val:.3f}")
    print()


def run_demo() -> None:
    segments = json.loads(_SAMPLE_PATH.read_text())
    pairs = _chunk_segments(segments)
    chunks = [p[0] for p in pairs]

    print(f"Demo transcript: {len(segments)} segments → {len(chunks)} chunks")
    print(f"Evaluating {len(QA_PAIRS)} QA pairs\n")

    bm25_results_per_q: list[list[int]] = []
    for q, _ in QA_PAIRS:
        bm25_results_per_q.append(_bm25_retrieve(q, chunks))

    def metrics(all_results: list[list[int]], label: str) -> None:
        h1 = sum(_hit_at_k(r, gt, 1) for r, (_, gt) in zip(all_results, QA_PAIRS)) / len(QA_PAIRS)
        h3 = sum(_hit_at_k(r, gt, 3) for r, (_, gt) in zip(all_results, QA_PAIRS)) / len(QA_PAIRS)
        h5 = sum(_hit_at_k(r, gt, 5) for r, (_, gt) in zip(all_results, QA_PAIRS)) / len(QA_PAIRS)
        mrr = sum(_reciprocal_rank(r, gt) for r, (_, gt) in zip(all_results, QA_PAIRS)) / len(QA_PAIRS)
        _print_table(label, [("Hit@1", h1), ("Hit@3", h3), ("Hit@5", h5), ("MRR", mrr)])

    metrics(bm25_results_per_q, "BM25")


def run_full() -> None:
    segments = json.loads(_SAMPLE_PATH.read_text())
    pairs = _chunk_segments(segments)
    chunks = [p[0] for p in pairs]

    print(f"Demo transcript: {len(segments)} segments → {len(chunks)} chunks")
    print(f"Evaluating {len(QA_PAIRS)} QA pairs\n")

    bm25_results: list[list[int]] = []
    sem_results: list[list[int]] = []
    hybrid_results: list[list[int]] = []

    for q, _ in QA_PAIRS:
        bm25_r = _bm25_retrieve(q, chunks)
        sem_r = _semantic_retrieve(q, chunks)
        hyb_r = _rrf(bm25_r, sem_r)
        bm25_results.append(bm25_r)
        sem_results.append(sem_r)
        hybrid_results.append(hyb_r)

    def metrics(all_results: list[list[int]], label: str) -> None:
        h1 = sum(_hit_at_k(r, gt, 1) for r, (_, gt) in zip(all_results, QA_PAIRS)) / len(QA_PAIRS)
        h3 = sum(_hit_at_k(r, gt, 3) for r, (_, gt) in zip(all_results, QA_PAIRS)) / len(QA_PAIRS)
        h5 = sum(_hit_at_k(r, gt, 5) for r, (_, gt) in zip(all_results, QA_PAIRS)) / len(QA_PAIRS)
        mrr = sum(_reciprocal_rank(r, gt) for r, (_, gt) in zip(all_results, QA_PAIRS)) / len(QA_PAIRS)
        _print_table(label, [("Hit@1", h1), ("Hit@3", h3), ("Hit@5", h5), ("MRR", mrr)])

    metrics(bm25_results, "BM25")
    metrics(sem_results, "Dense (all-MiniLM-L6-v2)")
    metrics(hybrid_results, "Hybrid BM25 + Dense (RRF)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoMind retrieval eval harness")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="BM25-only, no API keys required")
    group.add_argument("--full", action="store_true", help="BM25 + dense + hybrid (needs HF_TOKEN)")
    args = parser.parse_args()

    if args.full:
        run_full()
    else:
        run_demo()
