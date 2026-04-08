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

def add_chunks(video_id: str, chunks: list[str], embeddings: list[list[float]]):
    collection = get_collection()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{video_id}_{i}" for i in range(len(chunks))],
        metadatas=[{"video_id": video_id, "chunk_index": i} for i in range(len(chunks))]
    )

if __name__ == "__main__":
    col = get_collection()
    print(col.count())