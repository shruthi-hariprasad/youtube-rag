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

from backend.chunker import chunk_segments as _chunk_segments_impl

_SAMPLE_PATH = Path(__file__).parent / "sample_transcript.json"

# Hand-labelled QA pairs: (question, ground_truth_chunk_index)
# 9 chunks produced by the 300-word chunker with 50-word overlap:
# chunk 0: gradient descent definition, loss function, learning rate intro + "too large" warning
# chunk 1: learning rate range test, cosine annealing, batch/SGD/mini-batch variants
# chunk 2: backprop chain rule, forward/backward pass, vanishing gradient, Adam intro
# chunk 3: Adam beta hyperparameters, RMSprop, L1/L2 regularization, dropout
# chunk 4: dropout overlap + CNN intro, convolutional filters, stride/padding, max/avg pooling, receptive field
# chunk 5: residual networks, skip connections, degradation problem, transformer/self-attention, Q/K/V
# chunk 6: attention scores computation, softmax, multi-head attention, positional encodings, feed-forward sublayer
# chunk 7: batch normalization, scale/shift, internal covariate shift, layer normalization, early stopping
# chunk 8: data augmentation, transfer learning, fine-tuning, frozen layers
QA_PAIRS: list[tuple[str, int]] = [
    ("What is gradient descent?", 0),
    ("What does a loss function measure?", 0),
    ("How does the learning rate range test work?", 1),
    ("What is mini-batch gradient descent?", 1),
    ("How does backpropagation compute gradients using the chain rule?", 2),
    ("What is the vanishing gradient problem?", 2),
    ("What are the beta1 and beta2 hyperparameters in Adam?", 3),
    ("What is the difference between L1 and L2 regularization?", 3),
    ("What do convolutional filters detect in early layers?", 4),
    ("What is max pooling?", 4),
    ("How do residual networks solve the degradation problem?", 5),
    ("What are query, key, and value vectors in self-attention?", 5),
    ("How is the attention score between two tokens computed?", 6),
    ("What do positional encodings do in transformers?", 6),
    ("What is internal covariate shift?", 7),
    ("What is early stopping and how does the patience parameter work?", 7),
    ("How does data augmentation help prevent overfitting?", 8),
    ("What happens when fine-tuning too many layers on a small dataset?", 8),
]


_chunk_segments = _chunk_segments_impl


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
