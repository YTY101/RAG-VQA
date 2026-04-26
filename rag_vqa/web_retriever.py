from __future__ import annotations

from urllib.parse import quote

import requests

from .schemas import Evidence, QueryBundle


class WikipediaRetriever:
    """Small no-key web retriever for external encyclopedic evidence."""

    def __init__(self, timeout: int = 8, language: str = "zh") -> None:
        self.timeout = timeout
        self.language = language

    def retrieve(self, query: QueryBundle, top_k: int = 3) -> list[Evidence]:
        terms = query.keywords[:6] or [query.question]
        search_query = " ".join(terms)
        titles = self._search_titles(search_query, top_k=top_k)
        evidences: list[Evidence] = []
        for title in titles:
            summary = self._summary(title)
            if not summary:
                continue
            evidences.append(
                Evidence(
                    id=f"wiki:{title}",
                    title=title,
                    content=summary,
                    source=f"https://{self.language}.wikipedia.org/wiki/{quote(title)}",
                    type="web_text",
                    score=0.65,
                )
            )
        return evidences

    def _search_titles(self, search_query: str, top_k: int) -> list[str]:
        url = f"https://{self.language}.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": search_query,
            "limit": top_k,
            "namespace": 0,
            "format": "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return list(data[1]) if len(data) > 1 else []
        except Exception:
            return []

    def _summary(self, title: str) -> str | None:
        url = f"https://{self.language}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        try:
            resp = requests.get(url, timeout=self.timeout, headers={"accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            return data.get("extract")
        except Exception:
            return None

