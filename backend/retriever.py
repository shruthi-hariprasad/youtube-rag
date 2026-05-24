from .vector_store import get_collection
from .embedder import get_embeddings

def retrieve_chunks(
    query: str,
    video_ids: list[str] | None = None,
    n_results: int = 5
) -> list[dict]:
    
    collection = get_collection()
    
    # Embed the query the same way chunks were embedded
    query_embedding = get_embeddings([query])[0]
    
    # Build a filter if specific videos were requested
    where = {"video_id": {"$in": video_ids}} if video_ids else None
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "video_id": results["metadatas"][0][i]["video_id"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "start_time": results["metadatas"][0][i].get("start_time", 0.0),
            "distance": results["distances"][0][i],
        })

    return chunks