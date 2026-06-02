import os
import requests
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"


def search_web(query: str, max_results: int = 2) -> list[dict]:
    if not TAVILY_API_KEY:
        return []
    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [
            {
                "text": r.get("content", ""),
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source": "web",
            }
            for r in results
        ]
    except Exception:
        return []
