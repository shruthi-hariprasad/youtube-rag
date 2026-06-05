"""
RAGAS + LLM-as-judge evaluation harness.

Runs the full pipeline (retrieve → synthesize) on a pre-generated corpus
and scores faithfulness, answer relevancy, and context recall via RAGAS
using Groq as the judge LLM (no OpenAI needed).

Usage:
    pytest backend/eval/test_ragas_eval.py -m eval -v

    # Run a quick smoke test on 5 samples:
    pytest backend/eval/test_ragas_eval.py -m eval -v -k smoke

    # Run full 75-question suite:
    pytest backend/eval/test_ragas_eval.py -m eval -v -k full

CI note: corpus.json must be committed; DB + GROQ_API_KEY must be set.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextRecall,
    Faithfulness,
)

load_dotenv(Path(__file__).parent.parent / ".env")

CORPUS_PATH = Path(__file__).parent / "corpus.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

# Thresholds — fail CI if we fall below these
THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.65,
    "context_recall": 0.55,
}

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_corpus(n: int | None = None) -> list[dict]:
    if not CORPUS_PATH.exists():
        pytest.skip(f"corpus.json not found at {CORPUS_PATH}. Run generate_corpus.py first.")
    corpus = json.loads(CORPUS_PATH.read_text())
    if n:
        # Spread sample evenly across videos
        by_vid: dict[str, list[dict]] = {}
        for item in corpus:
            by_vid.setdefault(item["video_id"], []).append(item)
        result = []
        per_vid = max(1, n // len(by_vid))
        for items in by_vid.values():
            result.extend(items[:per_vid])
        return result[:n]
    return corpus


def _groq_with_retry(client, **kwargs) -> Any:
    """Call Groq with exponential backoff on rate limit errors."""
    from groq import RateLimitError
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait = 2 ** attempt * 15  # 15s, 30s, 60s, 120s, 240s
            print(f"\n  [rate limit] waiting {wait}s before retry {attempt+1}/5...")
            time.sleep(wait)
    raise RuntimeError("Groq rate limit persisted after 5 retries")


def _run_pipeline(question: str, video_id: str) -> tuple[str, list[str]]:
    """Run retrieve + synthesize, return (answer, list_of_context_strings)."""
    from backend.retriever import retrieve_chunks

    chunks = retrieve_chunks(question, video_ids=[video_id], n_results=8)
    contexts = [c["text"] for c in chunks]

    if not contexts:
        return "No relevant content found.", []

    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context_str = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(contexts[:6]))
    # Use 8b model for eval pipeline — preserves quota for the RAGAS judge calls
    resp = _groq_with_retry(
        client,
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question strictly based on the provided context. "
                    "Be concise. If the context does not contain the answer, say so."
                ),
            },
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
        ],
        max_tokens=512,
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip()
    return answer, contexts


def _build_ragas_dataset(samples: list[dict]) -> Dataset:
    rows: list[dict[str, Any]] = []
    total = len(samples)
    for i, item in enumerate(samples, 1):
        print(f"  [{i}/{total}] {item['question'][:60]}")
        answer, contexts = _run_pipeline(item["question"], item["video_id"])
        # Groq rate limit: small sleep between calls
        time.sleep(0.5)
        rows.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ragas_llm():
    from backend.eval.ragas_config import get_ragas_llm
    return get_ragas_llm()


@pytest.fixture(scope="module")
def ragas_embeddings():
    from backend.eval.ragas_config import get_ragas_embeddings
    return get_ragas_embeddings()


@pytest.mark.eval
@pytest.mark.smoke
def test_ragas_smoke(ragas_llm, ragas_embeddings):
    """Quick 8-sample sanity check — runs in ~2 minutes."""
    corpus = _load_corpus(n=8)
    dataset = _build_ragas_dataset(corpus)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextRecall(llm=ragas_llm),
    ]
    result = evaluate(dataset, metrics=metrics)
    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    print("\n=== RAGAS Smoke (n=8) ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    # Smoke test uses relaxed thresholds
    assert scores.get("faithfulness", 0) > 0.50, f"faithfulness too low: {scores.get('faithfulness')}"
    assert scores.get("answer_relevancy", 0) > 0.50, f"answer_relevancy too low"


@pytest.mark.eval
@pytest.mark.full
def test_ragas_full(ragas_llm, ragas_embeddings):
    """Full 75-question eval across all 4 video types. Saves results to eval_results.json."""
    corpus = _load_corpus()
    print(f"\nRunning full eval on {len(corpus)} samples across "
          f"{len({c['video_id'] for c in corpus})} videos...")

    dataset = _build_ragas_dataset(corpus)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextRecall(llm=ragas_llm),
    ]
    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()
    scores = df.mean(numeric_only=True).to_dict()

    # Per-video breakdown
    video_titles = {c["video_id"]: c["video_title"] for c in corpus}
    df["video_id"] = [corpus[i]["video_id"] for i in range(len(df))]
    per_video = df.groupby("video_id").mean(numeric_only=True).to_dict(orient="index")

    print("\n=== RAGAS Full Results ===")
    print(f"\n{'Metric':<25} {'Score':>6}")
    print("-" * 33)
    for k, v in scores.items():
        flag = " ✓" if v >= THRESHOLDS.get(k, 0) else " ✗ BELOW THRESHOLD"
        print(f"  {k:<23} {v:.3f}{flag}")

    print("\n--- Per-video breakdown ---")
    for vid_id, vid_scores in per_video.items():
        title = video_titles.get(vid_id, vid_id)[:45]
        print(f"\n  {title}")
        for k, v in vid_scores.items():
            print(f"    {k:<23} {v:.3f}")

    # Persist results for README table
    output = {
        "overall": scores,
        "per_video": {
            video_titles.get(vid_id, vid_id): s
            for vid_id, s in per_video.items()
        },
        "n_samples": len(corpus),
        "thresholds": THRESHOLDS,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")

    # Assert against thresholds
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0)
        assert score >= threshold, (
            f"{metric} = {score:.3f} is below threshold {threshold}. "
            f"Check eval_results.json for per-video breakdown."
        )


@pytest.mark.eval
@pytest.mark.retrieval
def test_retrieval_comparison():
    """
    Compare BM25 / Dense / Hybrid Hit@K and MRR on the corpus.
    This extends the existing eval_harness.py to cover all 4 video types.
    Prints a 3-way comparison table.
    """
    from backend.retriever import retrieve_chunks
    from backend.vector_store import get_chunks_for_videos
    from rank_bm25 import BM25Okapi
    from backend.embedder import get_embeddings
    import numpy as np

    corpus = _load_corpus()

    def hit_at_k(ranked_texts: list[str], ground_truth: str, k: int) -> bool:
        gt_lower = ground_truth.lower()
        for text in ranked_texts[:k]:
            # Check if any key phrases from ground truth appear in retrieved text
            gt_words = set(gt_lower.split())
            overlap = sum(1 for w in text.lower().split() if w in gt_words)
            if overlap / max(len(gt_words), 1) > 0.3:
                return True
        return False

    def mrr(ranked_texts: list[str], ground_truth: str) -> float:
        gt_lower = ground_truth.lower()
        gt_words = set(gt_lower.split())
        for rank, text in enumerate(ranked_texts, 1):
            overlap = sum(1 for w in text.lower().split() if w in gt_words)
            if overlap / max(len(gt_words), 1) > 0.3:
                return 1.0 / rank
        return 0.0

    results: dict[str, dict[str, list[float]]] = {
        "BM25": {"h1": [], "h3": [], "h5": [], "mrr": []},
        "Dense": {"h1": [], "h3": [], "h5": [], "mrr": []},
        "Hybrid": {"h1": [], "h3": [], "h5": [], "mrr": []},
    }

    # Group by video to avoid re-loading chunks
    by_video: dict[str, list[dict]] = {}
    for item in corpus:
        by_video.setdefault(item["video_id"], []).append(item)

    for vid_id, items in by_video.items():
        all_docs = get_chunks_for_videos([vid_id])
        if not all_docs:
            continue
        all_texts = [d["text"] for d in all_docs]
        tokenized = [t.lower().split() for t in all_texts]
        bm25 = BM25Okapi(tokenized)

        for item in items:
            q = item["question"]
            gt = item["ground_truth"]

            # BM25
            bm25_scores = bm25.get_scores(q.lower().split())
            bm25_ranked = [all_texts[i] for i in sorted(range(len(bm25_scores)),
                           key=lambda x: bm25_scores[x], reverse=True)]

            # Dense
            q_emb = np.array(get_embeddings([q])[0])
            doc_embs = np.array(get_embeddings(all_texts))
            cos_scores = doc_embs @ q_emb / (
                np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8
            )
            dense_ranked = [all_texts[i] for i in np.argsort(cos_scores)[::-1]]

            # Hybrid (retrieve_chunks uses RRF internally)
            hybrid_chunks = retrieve_chunks(q, video_ids=[vid_id], n_results=10)
            hybrid_ranked = [c["text"] for c in hybrid_chunks]

            for method, ranked in [("BM25", bm25_ranked), ("Dense", dense_ranked), ("Hybrid", hybrid_ranked)]:
                results[method]["h1"].append(float(hit_at_k(ranked, gt, 1)))
                results[method]["h3"].append(float(hit_at_k(ranked, gt, 3)))
                results[method]["h5"].append(float(hit_at_k(ranked, gt, 5)))
                results[method]["mrr"].append(mrr(ranked, gt))

            time.sleep(0.1)  # small pause between embedding calls

    print("\n=== Retrieval Comparison (BM25 / Dense / Hybrid) ===")
    print(f"\n{'Method':<12} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6}")
    print("-" * 42)
    for method, m in results.items():
        if not m["h1"]:
            continue
        n = len(m["h1"])
        h1 = sum(m["h1"]) / n
        h3 = sum(m["h3"]) / n
        h5 = sum(m["h5"]) / n
        mrr_avg = sum(m["mrr"]) / n
        print(f"  {method:<10} {h1:>6.3f} {h3:>6.3f} {h5:>6.3f} {mrr_avg:>6.3f}")
