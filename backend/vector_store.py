import os
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection():
    return client.get_or_create_collection(
        name="transcripts",
        metadata={"hnsw:space": "cosine"}
    )

def add_chunks(video_id: str, chunks: list[str], embeddings: list[list[float]], start_times: list[float] | None = None):
    collection = get_collection()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{video_id}_{i}" for i in range(len(chunks))],
        metadatas=[
            {
                "video_id": video_id,
                "chunk_index": i,
                "start_time": start_times[i] if start_times else 0.0,
            }
            for i in range(len(chunks))
        ]
    )

def get_chunks_for_videos(video_ids: list[str] | None = None) -> list[dict]:
    collection = get_collection()
    where = {"video_id": {"$in": video_ids}} if video_ids else None
    results = collection.get(where=where, include=["documents", "metadatas"])
    chunks = []
    for i in range(len(results["documents"])):
        chunks.append({
            "text": results["documents"][i],
            "video_id": results["metadatas"][i]["video_id"],
            "chunk_index": results["metadatas"][i]["chunk_index"],
            "start_time": results["metadatas"][i].get("start_time", 0.0),
            "distance": None,
        })
    return chunks


def delete_chunks(video_id: str):
    collection = get_collection()
    results = collection.get(where={"video_id": video_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])

if __name__ == "__main__":
    col = get_collection()
    print(col.count())