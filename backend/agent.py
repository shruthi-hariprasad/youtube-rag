import json
import logging
import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from .retriever import retrieve_chunks
from .embedder import get_embeddings
from .web_search import search_web as _search_web

load_dotenv()
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"       # 500k TPD free quota — used for all agent calls
DECISION_MODEL = "llama-3.1-8b-instant"

_WEB_DECISION_SYSTEM = """You have been given a question and transcript excerpts retrieved from the user's video library.
Decide whether web search is needed to give a complete answer.

Reply with valid JSON only, one of:
  {"web_needed": false}
  {"web_needed": true, "query": "<concise web search query>"}

Use web_needed=true only if the video excerpts clearly cannot answer the question.
If the excerpts contain a reasonable answer, use web_needed=false."""

_SYNTHESIZER_SYSTEM = """You are a helpful assistant. Answer strictly based on the provided sources.
- Do NOT include inline citations, source labels, or a title/heading at the start
- Do NOT invent, infer, or guess information not explicitly present in the sources
- If the sources list specific videos, only mention those exact videos — do not imply there are more
- Be concise and well-structured using markdown where helpful
- If the sources do not contain enough information to answer, say so honestly"""


def _err(msg: str):
    return f"data: {json.dumps({'type': 'token', 'token': msg})}\n\n"


def run_agent(question: str, video_ids: list[str], title_map: dict[str, str], meta_chunks: list[dict] | None = None):
    """Generator yielding SSE-formatted strings for the agent reasoning trace and answer."""
    _meta = list(meta_chunks or [])

    try:
        # For generic summary/overview questions, search using the video title
        # so retrieval finds topically relevant chunks instead of nothing
        _is_summary_q = any(w in question.lower() for w in ["summary", "summarize", "overview", "about", "what is this video", "what's this video"])
        titles_str = " ".join(title_map.values())
        search_query = titles_str if _is_summary_q and titles_str else question

        # Step 1: always search videos first
        yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'search_videos', 'query': search_query})}\n\n"
        video_chunks = retrieve_chunks(search_query, video_ids=video_ids)
        for c in video_chunks:
            c["title"] = title_map.get(c["video_id"], c["video_id"])
            c["source"] = "video"
        yield f"data: {json.dumps({'type': 'tool_result', 'tool': 'search_videos', 'count': len(video_chunks)})}\n\n"

        # Step 2: one LLM call to decide if web search is needed
        # Keep decision prompt small — just top 2 transcript chunks, no meta
        transcript_summary = "\n\n".join(f"[{c['title']}]\n{c['text'][:300]}" for c in video_chunks[:2])
        video_summary = transcript_summary.strip() or "(no results)"

        try:
            decision_response = client.chat.completions.create(
                model=DECISION_MODEL,
                messages=[
                    {"role": "system", "content": _WEB_DECISION_SYSTEM},
                    {"role": "user", "content": f"Question: {question}\n\nVideo excerpts:\n{video_summary}"},
                ],
                max_tokens=64,
            )
            raw = decision_response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            decision = json.loads(raw)
        except Exception:
            logger.exception("Web decision call failed, skipping web search")
            decision = {"web_needed": False}

        all_chunks: list[dict] = list(video_chunks)

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

        # Re-rank transcript/web chunks against the original question
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

        # Meta chunks always first — hold library-level facts transcript search can't surface
        final_chunks = _meta + unique_chunks

        if not final_chunks:
            yield _err("I could not find relevant information to answer your question.")
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
            return

        context = "\n\n".join(
            f"[{'Video' if c.get('source') == 'video' else 'Web'}: {c['title']}]\n{c['text'][:400]}"
            for c in final_chunks[:6]
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

        # One source per video (highest relevance), one per web URL
        # Only include if relevance is meaningful (>= 0.5)
        seen_source: set[str] = set()
        display_sources = []
        for c in unique_chunks:
            if c.get("_relevance", 0) < 0.5:
                continue
            key = c.get("video_id") or c.get("url", "")
            if key and key not in seen_source:
                seen_source.add(key)
                display_sources.append(c)
        yield f"data: {json.dumps({'type': 'done', 'sources': display_sources})}\n\n"

    except Exception as e:
        logger.exception("Agent pipeline error")
        if "rate_limit" in str(e).lower() or "429" in str(e):
            yield _err("The AI service is temporarily at capacity. Please try again in a few minutes.")
        else:
            yield _err("Something went wrong. Please try again.")
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
