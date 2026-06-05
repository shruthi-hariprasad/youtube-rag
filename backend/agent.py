import json
import logging
import os
from typing import Annotated
import operator

from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from .retriever import retrieve_chunks
from .web_search import search_web as _search_web

load_dotenv()
logger = logging.getLogger(__name__)

# LangSmith tracing — auto-instruments the LangGraph state graph when enabled.
# Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env to activate.
if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
    try:
        import langsmith  # noqa: F401 — import activates the tracing client
        os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "youtube-rag"))
        logger.info("LangSmith tracing enabled (project: %s)", os.getenv("LANGCHAIN_PROJECT"))
    except ImportError:
        logger.warning("LANGCHAIN_TRACING_V2=true but langsmith is not installed — pip install langsmith")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
DECISION_MODEL = "llama-3.1-8b-instant"

_QUERY_REWRITE_SYSTEM = """Given a conversation history and a follow-up question, rewrite the question into a single self-contained search query that includes all necessary context from the history.

Rules:
- Return ONLY the rewritten query — no explanation, no quotes, no punctuation at the end
- If the question is already self-contained, return it unchanged
- Keep the query concise (under 15 words)
- Include the key subject (person, event, topic) from history if the question omits it"""

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


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    # inputs
    question: str
    video_ids: list[str]
    title_map: dict[str, str]
    meta_chunks: list[dict]
    history: list[dict]
    # intermediate
    search_query: str
    video_chunks: list[dict]
    web_needed: bool
    web_query: str
    web_chunks: list[dict]
    unique_chunks: list[dict]
    final_chunks: list[dict]
    # SSE events emitted by each node — accumulated with list concatenation
    events: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def query_rewrite_node(state: PipelineState) -> dict:
    question = state["question"]
    history = state.get("history") or []
    search_query = question

    if len(history) >= 2:
        try:
            recent = history[-4:]
            history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
            rw = client.chat.completions.create(
                model=DECISION_MODEL,
                messages=[
                    {"role": "system", "content": _QUERY_REWRITE_SYSTEM},
                    {"role": "user", "content": f"History:\n{history_text}\n\nFollow-up question: {question}"},
                ],
                max_tokens=32,
            )
            rewritten = rw.choices[0].message.content.strip().strip('"').strip("'")
            if rewritten:
                search_query = rewritten
        except Exception:
            logger.exception("Query rewrite failed, using original question")

    return {"search_query": search_query}


def retrieve_node(state: PipelineState) -> dict:
    search_query = state["search_query"]
    video_ids = state["video_ids"]
    title_map = state["title_map"]

    events = [f"data: {json.dumps({'type': 'tool_call', 'tool': 'search_videos', 'query': search_query})}\n\n"]

    video_chunks = retrieve_chunks(search_query, video_ids=video_ids, n_results=8)
    for c in video_chunks:
        c["title"] = title_map.get(c["video_id"], c["video_id"])
        c["source"] = "video"

    events.append(f"data: {json.dumps({'type': 'tool_result', 'tool': 'search_videos', 'count': len(video_chunks)})}\n\n")

    return {"video_chunks": video_chunks, "events": events}


def web_decision_node(state: PipelineState) -> dict:
    search_query = state["search_query"]
    video_chunks = state["video_chunks"]

    transcript_summary = "\n\n".join(
        f"[{c['title']}]\n{c['text'][:500]}" for c in video_chunks[:5]
    )
    video_summary = transcript_summary.strip() or "(no results)"

    try:
        resp = client.chat.completions.create(
            model=DECISION_MODEL,
            messages=[
                {"role": "system", "content": _WEB_DECISION_SYSTEM},
                {"role": "user", "content": f"Question: {search_query}\n\nVideo excerpts:\n{video_summary}"},
            ],
            max_tokens=64,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        decision = json.loads(raw)
    except Exception:
        logger.exception("Web decision call failed, skipping web search")
        decision = {"web_needed": False}

    return {
        "web_needed": bool(decision.get("web_needed")),
        "web_query": decision.get("query", ""),
    }


def web_search_node(state: PipelineState) -> dict:
    web_query = state["web_query"]
    events = [f"data: {json.dumps({'type': 'tool_call', 'tool': 'search_web', 'query': web_query})}\n\n"]

    web_chunks = _search_web(web_query)

    events.append(f"data: {json.dumps({'type': 'tool_result', 'tool': 'search_web', 'count': len(web_chunks)})}\n\n")

    return {"web_chunks": web_chunks, "events": events}


def merge_chunks_node(state: PipelineState) -> dict:
    """Deduplicate + rank chunks and pin relevant meta chunks."""
    video_chunks = state["video_chunks"]
    web_chunks = state.get("web_chunks") or []
    meta_chunks = state.get("meta_chunks") or []
    meta_by_vid = {m["video_id"]: m for m in meta_chunks}

    all_chunks = list(video_chunks) + list(web_chunks)

    seen: set[str] = set()
    unique_chunks: list[dict] = []
    for c in all_chunks:
        key = f"{c.get('video_id', '')}:{c.get('chunk_index', c.get('url', c.get('text', '')[:60]))}"
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    for rank, c in enumerate(unique_chunks):
        if "_relevance" not in c:
            c["_relevance"] = max(0.0, 1.0 - rank * 0.08)

    retrieved_video_ids = {c.get("video_id") for c in unique_chunks}
    relevant_meta = [meta_by_vid[vid] for vid in retrieved_video_ids if vid in meta_by_vid]
    final_chunks = relevant_meta + unique_chunks

    return {"unique_chunks": unique_chunks, "final_chunks": final_chunks}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_web(state: PipelineState) -> str:
    if state.get("web_needed") and state.get("web_query"):
        return "web_search"
    return "merge_chunks"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("query_rewrite", query_rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("web_decision", web_decision_node)
    g.add_node("web_search", web_search_node)
    g.add_node("merge_chunks", merge_chunks_node)

    g.set_entry_point("query_rewrite")
    g.add_edge("query_rewrite", "retrieve")
    g.add_edge("retrieve", "web_decision")
    g.add_conditional_edges("web_decision", _route_web, {
        "web_search": "web_search",
        "merge_chunks": "merge_chunks",
    })
    g.add_edge("web_search", "merge_chunks")
    g.add_edge("merge_chunks", END)

    return g.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public interface — preserves the SSE generator contract for FastAPI
# ---------------------------------------------------------------------------

def _err(msg: str) -> str:
    return f"data: {json.dumps({'type': 'token', 'token': msg})}\n\n"


def run_agent(
    question: str,
    video_ids: list[str],
    title_map: dict[str, str],
    meta_chunks: list[dict] | None = None,
    history: list[dict] | None = None,
):
    """Generator yielding SSE-formatted strings for the agent reasoning trace and answer.

    Internally runs a LangGraph state graph for the first four pipeline stages
    (query_rewrite → retrieve → web_decision → merge_chunks), streaming events
    from each node as it completes. Token streaming for synthesis is handled
    directly to preserve true per-token latency.
    """
    try:
        initial_state: PipelineState = {
            "question": question,
            "video_ids": video_ids,
            "title_map": title_map,
            "meta_chunks": meta_chunks or [],
            "history": history or [],
            "search_query": question,
            "video_chunks": [],
            "web_needed": False,
            "web_query": "",
            "web_chunks": [],
            "unique_chunks": [],
            "final_chunks": [],
            "events": [],
        }

        # Stream node-by-node: yield each node's SSE events as soon as the node finishes
        final_state: PipelineState | None = None
        for chunk in _graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                for event in node_update.get("events", []):
                    yield event
            # Merge updates into final_state manually for the last snapshot
            if final_state is None:
                final_state = dict(initial_state)
            for node_update in chunk.values():
                final_state.update(node_update)

        if final_state is None:
            yield _err("Pipeline produced no output.")
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
            return

        final_chunks: list[dict] = final_state.get("final_chunks") or []
        unique_chunks: list[dict] = final_state.get("unique_chunks") or []

        if not final_chunks:
            yield _err("I could not find relevant information to answer your question.")
            yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
            return

        # Synthesize — streaming tokens directly (bypasses graph to preserve per-token latency)
        context = "\n\n".join(
            f"[{'Video' if c.get('source') == 'video' else 'Web'}: {c['title']}]\n{c['text'][:1500]}"
            for c in final_chunks[:10]
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

        # Build display sources
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

        yield f"data: {json.dumps({'type': 'done', 'sources': display_sources})}\n\n"

    except Exception as e:
        logger.exception("Agent pipeline error")
        if "rate_limit" in str(e).lower() or "429" in str(e):
            yield _err("The AI service is temporarily at capacity. Please try again in a few minutes.")
        else:
            yield _err("Something went wrong. Please try again.")
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
