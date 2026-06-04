import json
import logging
import os
from groq import Groq
from dotenv import load_dotenv
from .retriever import retrieve_chunks
from .web_search import search_web as _search_web

load_dotenv()
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"       # synthesis
DECISION_MODEL = "llama-3.1-8b-instant" # web decision only (separate quota)

_WEB_DECISION_SYSTEM = """You have been given a question and transcript excerpts from the user's video library.
Decide whether a web search would meaningfully help answer the question.

Reply with valid JSON only — no other text:
  {"web_needed": false}
  {"web_needed": true, "query": "<concise web search query>"}

Use web_needed=true when ANY of these apply:
- The question asks about facts, events, or entities not present in the excerpts
- The question asks about things outside the scope of the video (e.g. other events, other people, external context)
- The excerpts only partially answer the question and external facts would help complete it
- The question is clearly about current events, rankings, schedules, or results

Use web_needed=false only when the excerpts already contain a complete answer to the question."""

_SYNTHESIZER_SYSTEM = """You are a helpful assistant. Answer strictly based on the provided sources.
- Do NOT include inline citations, source labels, or a title/heading at the start
- Do NOT invent, infer, or guess information not explicitly present in the sources
- If the sources list specific videos, only mention those exact videos — do not imply there are more
- Be concise and well-structured using markdown where helpful
- If the sources do not contain enough information to answer, say so honestly"""


def _err(msg: str):
    return f"data: {json.dumps({'type': 'token', 'token': msg})}\n\n"


def run_agent(
    question: str,
    video_ids: list[str],
    title_map: dict[str, str],
    meta_chunks: list[dict] | None = None,
    history: list[dict] | None = None,
):
    """Generator yielding SSE-formatted strings for the agent reasoning trace and answer."""
    _meta_by_vid = {m["video_id"]: m for m in (meta_chunks or [])}

    try:
        # Step 1: always retrieve video chunks first (retriever handles hybrid BM25+cosine+RRF)
        yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'search_videos', 'query': question})}\n\n"
        video_chunks = retrieve_chunks(question, video_ids=video_ids)
        for c in video_chunks:
            c["title"] = title_map.get(c["video_id"], c["video_id"])
            c["source"] = "video"
        yield f"data: {json.dumps({'type': 'tool_result', 'tool': 'search_videos', 'count': len(video_chunks)})}\n\n"

        # Step 2: one cheap LLM call to decide if web search adds value
        # Show all retrieved chunks (up to 5) so the model has full context to decide
        transcript_summary = "\n\n".join(
            f"[{c['title']}]\n{c['text'][:500]}" for c in video_chunks[:5]
        )
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

        # Retriever already ranks by hybrid BM25+cosine — trust that order.
        # Assign a synthetic relevance score based on retrieval rank for source filtering later.
        for rank, c in enumerate(unique_chunks):
            if "_relevance" not in c:
                # Linear decay: rank 0 → 1.0, rank 4 → 0.6, beyond that → below threshold
                c["_relevance"] = max(0.0, 1.0 - rank * 0.08)

        # Pin meta chunks only for videos that actually appear in retrieved content.
        # This prevents meta chunks from consuming all context slots on library queries.
        retrieved_video_ids = {c.get("video_id") for c in unique_chunks}
        relevant_meta = [_meta_by_vid[vid] for vid in retrieved_video_ids if vid in _meta_by_vid]
        final_chunks = relevant_meta + unique_chunks

        if not final_chunks:
            yield _err("I could not find relevant information to answer your question.")
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
            return

        # Use fuller chunk text (chunks are ~300 words; 1500 chars ≈ 250 words — better coverage)
        context = "\n\n".join(
            f"[{'Video' if c.get('source') == 'video' else 'Web'}: {c['title']}]\n{c['text'][:1500]}"
            for c in final_chunks[:8]
        )

        synth_messages = [{"role": "system", "content": _SYNTHESIZER_SYSTEM}]
        if history:
            synth_messages.extend(history[-4:])
        synth_messages.append(
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        )

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

        # Up to 3 chunks per video (sorted by timestamp so the UI shows them in order),
        # one entry per web URL. Only include if relevance is meaningful (>= 0.5).
        video_source_counts: dict[str, int] = {}
        web_seen: set[str] = set()
        display_sources = []
        for c in unique_chunks:
            if c.get("_relevance", 0) < 0.5:
                continue
            if c.get("source") == "web":
                url = c.get("url", "")
                if url and url not in web_seen:
                    web_seen.add(url)
                    display_sources.append(c)
            else:
                vid = c.get("video_id", "")
                if vid and video_source_counts.get(vid, 0) < 3:
                    video_source_counts[vid] = video_source_counts.get(vid, 0) + 1
                    display_sources.append(c)

        # Sort video sources by timestamp so they appear in chronological order
        display_sources.sort(key=lambda c: (
            c.get("source") == "web",       # video sources first
            c.get("video_id", ""),           # group by video
            c.get("start_time") or 0.0,      # then chronological
        ))
        yield f"data: {json.dumps({'type': 'done', 'sources': display_sources})}\n\n"

    except Exception as e:
        logger.exception("Agent pipeline error")
        if "rate_limit" in str(e).lower() or "429" in str(e):
            yield _err("The AI service is temporarily at capacity. Please try again in a few minutes.")
        else:
            yield _err("Something went wrong. Please try again.")
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
