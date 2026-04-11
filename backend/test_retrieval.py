from .retriever import retrieve_chunks

query = "What products did they rate?"
results = retrieve_chunks(query)

for r in results:
    print(f"\nDistance: {r['distance']:.3f}")
    print(f"Chunk: {r['text'][:200]}...")