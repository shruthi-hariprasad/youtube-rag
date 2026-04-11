import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query: str, chunks: list[dict]) -> dict:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source: {chunk['title']}]\n{chunk['text']}\n"

    prompt = f"""You are a helpful assistant that answers questions based on YouTube video transcripts.

Here are the most relevant parts of the transcript:
{context}

Question: {query}

Answer based only on the transcript content above. If the answer is not in the transcript, say so clearly. Be specific and concise."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    sources = list(set([chunk["title"] for chunk in chunks]))

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }