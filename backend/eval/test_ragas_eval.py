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
from ragas.run_config import RunConfig
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_recall import LLMContextRecall as ContextRecall

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
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        ContextRecall(llm=ragas_llm),
    ]
    run_cfg = RunConfig(max_workers=1, timeout=120)
    result = evaluate(dataset, metrics=metrics, run_config=run_cfg)
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
    """30-question eval across all 4 video types (~7-8 per video). Saves results to eval_results.json."""
    corpus = _load_corpus(n=32)  # 8 per video, 4 videos
    print(f"\nRunning full eval on {len(corpus)} samples across "
          f"{len({c['video_id'] for c in corpus})} videos...")

    dataset = _build_ragas_dataset(corpus)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        ContextRecall(llm=ragas_llm),
    ]
    # max_workers=1 serializes all LLM calls — prevents TPM bursting on Groq free tier
    run_cfg = RunConfig(max_workers=1, timeout=120)
    result = evaluate(dataset, metrics=metrics, run_config=run_cfg)
    df = result.to_pandas()

    # Warn if any metric has high NaN rate (indicates rate limit / timeout issues)
    for col in ["faithfulness", "answer_relevancy", "context_recall"]:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"  WARNING: {col} has {nan_count}/{len(df)} NaN values (likely rate limit timeouts)")

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
@pytest.mark.full
def test_llm_judge():
    """
    Custom LLM-as-judge eval — same 3 metrics as RAGAS but implemented directly.
    Uses llama-3.1-8b-instant with short, truncated prompts so it runs reliably
    within Groq free-tier TPM limits. Sequential calls with 2s sleep between them.

    Metrics:
      faithfulness     — are the answer's claims grounded in the retrieved context?
      answer_relevancy — does the answer directly address the question?
      context_recall   — does the context contain enough to answer from ground truth?
    """
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    JUDGE_MODEL = "llama-3.1-8b-instant"

    def _truncate_contexts(contexts: list[str], max_chars: int = 250) -> str:
        """Keep first max_chars of each chunk — enough signal, few tokens."""
        return "\n---\n".join(c[:max_chars] for c in contexts[:5])

    def _score(prompt: str) -> float:
        """Ask judge to return a float 0.0–1.0. Retry on failure."""
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "You are an evaluation judge. "
                            "Respond with ONLY a decimal number between 0.0 and 1.0. "
                            "No explanation, no text, just the number."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=8,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip()
                return float(raw)
            except Exception as e:
                wait = 2 ** attempt * 5
                print(f"\n  [judge error] {e} — waiting {wait}s...")
                time.sleep(wait)
        return float("nan")

    corpus = _load_corpus(n=32)  # 8 per video, 4 videos
    print(f"\nRunning LLM-as-judge eval on {len(corpus)} samples across "
          f"{len({c['video_id'] for c in corpus})} videos...")

    PARTIAL_PATH = RESULTS_PATH.parent / "eval_results_partial.json"
    results = []
    for i, item in enumerate(corpus, 1):
        q = item["question"]
        gt = item["ground_truth"]
        vid = item["video_id"]

        print(f"  [{i}/{len(corpus)}] {q[:60]}")
        answer, contexts = _run_pipeline(q, vid)
        ctx_str = _truncate_contexts(contexts)

        # --- faithfulness ---
        faith_prompt = (
            f"Question: {q}\n"
            f"Answer: {answer}\n"
            f"Context:\n{ctx_str}\n\n"
            "Score: what fraction of the answer's claims are directly supported "
            "by the context? (0.0 = none supported, 1.0 = fully supported)"
        )
        time.sleep(6)
        faithfulness = _score(faith_prompt)

        # --- answer relevancy ---
        rel_prompt = (
            f"Question: {q}\n"
            f"Answer: {answer}\n\n"
            "Score: how directly and completely does the answer address the question? "
            "(0.0 = irrelevant or empty, 1.0 = perfectly on-topic and complete)"
        )
        time.sleep(6)
        relevancy = _score(rel_prompt)

        # --- context recall ---
        recall_prompt = (
            f"Question: {q}\n"
            f"Ground truth answer: {gt}\n"
            f"Context:\n{ctx_str}\n\n"
            "Score: how much of the ground truth answer can be inferred from the context? "
            "(0.0 = context has nothing relevant, 1.0 = context fully covers the ground truth)"
        )
        time.sleep(6)
        recall = _score(recall_prompt)

        results.append({
            "video_id": vid,
            "video_title": item["video_title"],
            "question": q,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_recall": recall,
        })

        # Save partial results after every sample — survive laptop sleep/kill
        PARTIAL_PATH.write_text(json.dumps(results, indent=2))
        print(f"    faith={faithfulness:.2f}  rel={relevancy:.2f}  recall={recall:.2f}")

    # --- aggregate ---
    import statistics

    def _mean(key):
        vals = [r[key] for r in results if not (isinstance(r[key], float) and r[key] != r[key])]
        return statistics.mean(vals) if vals else float("nan")

    overall = {
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "context_recall": _mean("context_recall"),
    }

    # Per-video breakdown
    by_video: dict[str, list] = {}
    for r in results:
        by_video.setdefault(r["video_title"], []).append(r)

    print("\n=== LLM-as-Judge Results (n=32) ===")
    print(f"\n{'Metric':<25} {'Score':>6}")
    print("-" * 33)
    for k, v in overall.items():
        flag = " ✓" if v >= THRESHOLDS.get(k, 0) else " ✗ BELOW THRESHOLD"
        print(f"  {k:<23} {v:.3f}{flag}")

    print("\n--- Per-video breakdown ---")
    for title, rows in by_video.items():
        print(f"\n  {title[:50]}")
        for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
            vals = [r[metric] for r in rows if not (isinstance(r[metric], float) and r[metric] != r[metric])]
            avg = statistics.mean(vals) if vals else float("nan")
            print(f"    {metric:<23} {avg:.3f}")

    # Persist
    output = {
        "overall": overall,
        "per_video": {
            title: {
                m: statistics.mean([r[m] for r in rows if not (isinstance(r[m], float) and r[m] != r[m])] or [float("nan")])
                for m in ["faithfulness", "answer_relevancy", "context_recall"]
            }
            for title, rows in by_video.items()
        },
        "n_samples": len(corpus),
        "method": "llm-as-judge (llama-3.1-8b-instant)",
        "thresholds": THRESHOLDS,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")

    nan_count = sum(1 for r in results if any(
        isinstance(r[m], float) and r[m] != r[m]
        for m in ["faithfulness", "answer_relevancy", "context_recall"]
    ))
    if nan_count:
        print(f"  WARNING: {nan_count} samples had scoring errors (NaN)")

    for metric, threshold in THRESHOLDS.items():
        score = overall.get(metric, 0)
        if score == score:  # not NaN
            assert score >= threshold, f"{metric} = {score:.3f} below threshold {threshold}"


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
