from rank_bm25 import BM25Okapi

from .vector_store import get_collection, get_chunks_for_videos
from .embedder import get_embeddings

DISTANCE_THRESHOLD = 0.8


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def retrieve_chunks(
    query: str,
    video_ids: list[str] | None = None,
    n_results: int = 5,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> list[dict]:
    collection = get_collection()
    query_embedding = get_embeddings([query])[0]
    where = {"video_id": {"$in": video_ids}} if video_ids else None

    # --- Semantic search: fetch a wider candidate pool for RRF ---
    semantic_n = min(n_results * 4, 20)
    try:
        sem_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=semantic_n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        sem_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    # Build semantic pool, filtering by distance threshold
    semantic_pool: list[dict] = []
    for i in range(len(sem_results["documents"][0])):
        dist = sem_results["distances"][0][i]
        if dist <= distance_threshold:
            semantic_pool.append({
                "text": sem_results["documents"][0][i],
                "video_id": sem_results["metadatas"][0][i]["video_id"],
                "chunk_index": sem_results["metadatas"][0][i]["chunk_index"],
                "start_time": sem_results["metadatas"][0][i].get("start_time", 0.0),
                "distance": dist,
            })

    # --- BM25 search over all chunks for the relevant videos ---
    all_docs = get_chunks_for_videos(video_ids)
    bm25_pool: list[dict] = []
    if all_docs:
        tokenized_corpus = [d["text"].lower().split() for d in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_pool = [all_docs[i] for i in top_indices[:semantic_n] if bm25_scores[i] > 0]

    # --- RRF fusion ---
    fused: dict[str, dict] = {}

    for rank, chunk in enumerate(semantic_pool):
        key = f"{chunk['video_id']}_{chunk['chunk_index']}"
        chunk["rrf_score"] = _rrf_score(rank)
        fused[key] = chunk

    for rank, chunk in enumerate(bm25_pool):
        key = f"{chunk['video_id']}_{chunk['chunk_index']}"
        if key in fused:
            fused[key]["rrf_score"] += _rrf_score(rank)
        else:
            chunk["rrf_score"] = _rrf_score(rank)
            fused[key] = chunk

    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:n_results]
