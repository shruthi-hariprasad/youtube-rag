"""
Configure RAGAS to use Groq (llama-3.3-70b) as the judge LLM
and our existing HuggingFace embedder for embeddings.
Avoids any OpenAI dependency.
"""
from __future__ import annotations

import os
import warnings

from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings


class _HFEmbeddings(BaseRagasEmbeddings):
    """Thin wrapper around our existing embedder for RAGAS."""

    def embed_query(self, text: str) -> list[float]:
        from backend.embedder import get_embeddings
        return get_embeddings([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from backend.embedder import get_embeddings
        return get_embeddings(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def get_ragas_llm() -> LangchainLLMWrapper:
    # LangchainLLMWrapper is the only way to use non-OpenAI LLMs with RAGAS 0.4.x.
    # The deprecation warning fires at import but the class still functions correctly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(
            ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0,
            )
        )


def get_ragas_embeddings() -> _HFEmbeddings:
    return _HFEmbeddings()
