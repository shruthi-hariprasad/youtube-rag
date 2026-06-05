"""
MCP (Model Context Protocol) server — JSON-RPC 2.0 over HTTP.

Exposes the retrieval pipeline as a callable tool so any MCP-compatible
client (Claude Desktop, Cursor, etc.) can search your video library.

Mounted at /mcp in main.py.

Tool exposed:
  search_video_library(query, video_ids?, n_results?) → ranked chunks

JSON-RPC methods supported:
  initialize        — handshake, returns server capabilities
  tools/list        — lists available tools with their input schemas
  tools/call        — executes a tool and returns the result
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .retriever import retrieve_chunks

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "search_video_library",
        "description": (
            "Search a user's YouTube video library using hybrid BM25 + dense + RRF retrieval. "
            "Returns the most relevant transcript chunks ranked by relevance. "
            "Use this to answer questions about specific videos or across an entire library."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question to retrieve relevant chunks for.",
                },
                "video_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of YouTube video IDs to restrict search to. Omit to search all videos.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of chunks to return (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _call_tool(name: str, arguments: dict) -> dict:
    if name != "search_video_library":
        return {"error": f"Unknown tool: {name}"}

    query = arguments.get("query", "").strip()
    if not query:
        return {"error": "query is required"}

    video_ids = arguments.get("video_ids") or None
    n_results = min(int(arguments.get("n_results", 5)), 20)

    chunks = retrieve_chunks(query, video_ids=video_ids, n_results=n_results)

    results = [
        {
            "video_id": c["video_id"],
            "text": c["text"],
            "start_time": c.get("start_time", 0.0),
            "chunk_index": c.get("chunk_index", 0),
            "rrf_score": round(c.get("rrf_score", 0.0), 4),
        }
        for c in chunks
    ]

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps({"query": query, "chunks": results}, indent=2),
            }
        ]
    }


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 dispatch
# ---------------------------------------------------------------------------

def _rpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def _rpc_ok(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _dispatch(method: str, params: dict, id: Any) -> dict:
    if method == "initialize":
        return _rpc_ok(id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "youtube-rag", "version": "1.0.0"},
        })

    if method == "tools/list":
        return _rpc_ok(id, {"tools": _TOOLS})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not tool_name:
            return _rpc_error(id, -32602, "Missing tool name")
        try:
            result = _call_tool(tool_name, arguments)
        except Exception as e:
            logger.exception("Tool call failed: %s", tool_name)
            return _rpc_error(id, -32603, f"Tool execution error: {e}")
        return _rpc_ok(id, result)

    return _rpc_error(id, -32601, f"Method not found: {method}")


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

@router.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            _rpc_error(None, -32700, "Parse error: invalid JSON"),
            status_code=400,
        )

    # Support both single requests and batches
    if isinstance(body, list):
        responses = [_dispatch(r.get("method", ""), r.get("params", {}), r.get("id")) for r in body]
        return JSONResponse(responses)

    method = body.get("method", "")
    params = body.get("params", {})
    rpc_id = body.get("id")

    return JSONResponse(_dispatch(method, params, rpc_id))
