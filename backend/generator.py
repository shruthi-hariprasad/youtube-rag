import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on YouTube video transcript excerpts provided to you.

Rules:
- Answer ONLY using information from the provided transcript excerpts
- When referencing information, cite which video it came from using the [Source: ...] labels
- If the answer is not in the transcripts, say exactly: "I couldn't find that information in these transcripts."
- Never speculate, hallucinate, or draw on knowledge outside the provided excerpts
- Be specific, accurate, and concise"""


def _build_messages(query: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[Source: {chunk['title']}]\n{chunk['text']}" for chunk in chunks
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript excerpts:\n{context}\n\nQuestion: {query}"},
    ]


def generate_answer(query: str, chunks: list[dict]) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=_build_messages(query, chunks),
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }


def stream_answer(query: str, chunks: list[dict]):
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=_build_messages(query, chunks),
        stream=True,
    )
    for piece in stream:
        token = piece.choices[0].delta.content
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"
    yield f"data: {json.dumps({'sources': chunks, 'done': True})}\n\n"
