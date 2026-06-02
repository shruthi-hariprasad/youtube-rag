import json
import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from .retriever import retrieve_chunks
from .embedder import get_embeddings
from .web_search import search_web as _search_web

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

_WEB_DECISION_SYSTEM = """You have been given a question and transcript excerpts retrieved from the user's video library.
Decide whether web search is needed to give a complete answer.

Reply with valid JSON only, one of:
  {{"web_needed": false}}
  {{"web_needed": true, "query": "<concise web search query>"}}

Use web_needed=true only if the video excerpts clearly cannot answer the question.
If the excerpts contain a reasonable answer, use web_needed=false."""

_SYNTHESIZER_SYSTEM = """You are a helpful assistant. Synthesize the provided sources into a clear, accurate answer.
- Do NOT include inline citations, source labels, or a title/heading at the start
- If sources conflict, note it briefly
- Be concise and well-structured using markdown where helpful
- If no sources contain relevant information, say so honestly"""


def run_agent(question: str, video_ids: list[str], title_map: dict[str, str], meta_chunks: list[dict] | None = None):
    """Generator yielding SSE-formatted strings for the agent reasoning trace and answer."""

    # Step 1: always search videos first (no LLM needed for this decision)
    yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'search_videos', 'query': question})}\n\n"
    video_chunks = retrieve_chunks(question, video_ids=video_ids)
    for c in video_chunks:
        c["title"] = title_map.get(c["video_id"], c["video_id"])
        c["source"] = "video"
    yield f"data: {json.dumps({'type': 'tool_result', 'tool': 'search_videos', 'count': len(video_chunks)})}\n\n"

    # Prepend metadata chunks (title, channel) so the synthesizer can answer
    # questions about the video itself that won't appear in transcript text
    all_video_chunks = list(meta_chunks or []) + video_chunks

    # Step 2: one LLM call to decide if web search is needed
    video_summary = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in all_video_chunks[:4]) or "(no results)"
    decision_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _WEB_DECISION_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nVideo excerpts:\n{video_summary}"},
        ],
        max_tokens=64,
    )
    raw = decision_response.choices[0].message.content.strip()
    # strip markdown code fences if model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        decision = json.loads(raw)
    except Exception:
        decision = {"web_needed": False}

    all_chunks: list[dict] = list(all_video_chunks)

    # Step 3: optionally search the web
    if decision.get("web_needed") and decision.get("query"):
        web_query = decision["query"]
        yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'search_web', 'query': web_query})}\n\n"
        web_results = _search_web(web_query)
        all_chunks.extend(web_results)
        yield f"data: {json.dumps({'type': 'tool_result', 'tool': 'search_web', 'count': len(web_results)})}\n\n"

    # Deduplicate by chunk identity
    seen: set[str] = set()
    unique_chunks: list[dict] = []
    for c in all_chunks:
        key = f"{c.get('video_id', '')}:{c.get('chunk_index', c.get('url', c.get('text', '')[:60]))}"
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    # Re-rank against the original question (single batched embedding call)
    if unique_chunks:
        texts = [c.get("text", "") for c in unique_chunks]
        all_embs = np.array(get_embeddings([question] + texts))
        q_emb = all_embs[0]
        chunk_embs = all_embs[1:]
        q_norm = np.linalg.norm(q_emb)
        for c, c_emb in zip(unique_chunks, chunk_embs):
            norm = q_norm * np.linalg.norm(c_emb)
            c["_relevance"] = float(np.dot(q_emb, c_emb) / norm) if norm > 0 else 0.0
        unique_chunks.sort(key=lambda c: c["_relevance"], reverse=True)

    if not unique_chunks:
        yield f"data: {json.dumps({'type': 'token', 'token': 'I could not find relevant information to answer your question.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
        return

    context = "\n\n".join(
        f"[{'Video' if c.get('source') == 'video' else 'Web'}: {c['title']}]\n{c['text']}"
        for c in unique_chunks
    )

    synth_messages = [
        {"role": "system", "content": _SYNTHESIZER_SYSTEM},
        {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
    ]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=synth_messages,
        max_tokens=1024,
        stream=True,
    )

    for piece in stream:
        token = piece.choices[0].delta.content
        if token:
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    yield f"data: {json.dumps({'type': 'done', 'sources': unique_chunks})}\n\n"
