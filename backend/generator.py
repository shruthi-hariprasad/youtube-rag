import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"  # 500k TPD free quota

def _system_prompt(multi_video: bool) -> str:
    source_rule = (
        "- When referencing information from different videos, note which video it came from using the [Source: ...] labels"
        if multi_video else
        "- Do NOT include source labels or citations inline — the UI shows sources separately"
    )
    return f"""You are a helpful assistant that answers questions strictly based on YouTube video transcript excerpts provided to you.

Rules:
- Answer ONLY using information from the provided transcript excerpts
{source_rule}
- If the answer is not in the transcripts, say exactly: "I couldn't find that information in these transcripts."
- Never speculate, hallucinate, or draw on knowledge outside the provided excerpts
- Do NOT start your answer with a title or heading — answer directly
- Be specific, accurate, and concise
- Use markdown formatting (bullet points, bold) where it improves readability"""


def _build_messages(query: str, chunks: list[dict], history: list[dict] | None = None) -> list[dict]:
    unique_videos = len(set(c.get("video_id", "") for c in chunks))
    multi_video = unique_videos > 1
    context = "\n\n".join(
        f"[Source: {chunk['title']}]\n{chunk['text']}" for chunk in chunks
    )
    messages = [{"role": "system", "content": _system_prompt(multi_video)}]
    if history:
        messages.extend(history[-6:])
    messages.append({
        "role": "user",
        "content": f"Transcript excerpts:\n{context}\n\nQuestion: {query}",
    })
    return messages


def generate_answer(query: str, chunks: list[dict], history: list[dict] | None = None) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=_build_messages(query, chunks, history),
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }


def stream_answer(query: str, chunks: list[dict], history: list[dict] | None = None):
    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=_build_messages(query, chunks, history),
        stream=True,
    )
    for piece in stream:
        token = piece.choices[0].delta.content
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"
    yield f"data: {json.dumps({'sources': chunks, 'done': True})}\n\n"


def generate_summary_and_questions(transcript_text: str) -> dict:
    # Sample beginning, middle, and end so long videos aren't summarised from intro only
    length = len(transcript_text)
    if length <= 6000:
        excerpt = transcript_text
    else:
        chunk = 2000
        mid = length // 2
        excerpt = (
            transcript_text[:chunk]
            + "\n...\n"
            + transcript_text[mid - chunk // 2 : mid + chunk // 2]
            + "\n...\n"
            + transcript_text[-chunk:]
        )
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond with valid JSON only, no other text.",
            },
            {
                "role": "user",
                "content": (
                    "Analyze this YouTube transcript and return a JSON object with exactly two keys:\n"
                    '- "summary": a 2-sentence overview of the video\n'
                    '- "questions": an array of exactly 3 specific questions a viewer might ask\n\n'
                    f"Transcript:\n{excerpt}\n\nJSON:"
                ),
            },
        ],
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1].lstrip("json").strip() if len(parts) > 1 else content
    return json.loads(content)
