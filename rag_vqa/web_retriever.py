from __future__ import annotations

import re
from urllib.parse import quote

import requests

from .config import Settings
from .debug import debug_dump
from .schemas import Evidence, QueryBundle


class WikipediaRetriever:
    """Small no-key web retriever for external encyclopedic evidence."""

    def __init__(
        self,
        timeout: int = 8,
        language: str = "zh",
        settings: Settings | None = None,
        use_env_proxy: bool = False,
    ) -> None:
        self.timeout = timeout
        self.language = language
        self.settings = settings or Settings()
        self.session = requests.Session()
        self.session.trust_env = use_env_proxy
        self.session.headers.update(
            {
                "User-Agent": "RAG-VQA/0.1 (educational project; wikipedia retrieval)",
                "Accept": "application/json",
            }
        )

    def retrieve(self, query: QueryBundle, top_k: int = 3) -> list[Evidence]:
        search_query = self._build_search_query(query)
        debug_dump(
            self.settings,
            "web.retrieve.start",
            {
                "language": self.language,
                "top_k": top_k,
                "search_query": search_query,
                "trust_env_proxy": self.session.trust_env,
            },
        )
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
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "srlimit": top_k,
            "format": "json",
        }
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            return [str(item.get("title", "")).strip() for item in results if str(item.get("title", "")).strip()]
        except Exception as exc:
            debug_dump(
                self.settings,
                "web.search.error",
                {"url": url, "search_query": search_query, "error": repr(exc)},
            )
            return []

    def _summary(self, title: str) -> str | None:
        url = f"https://{self.language}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        try:
            resp = self.session.get(url, timeout=self.timeout, headers={"accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            return data.get("extract")
        except Exception as exc:
            debug_dump(
                self.settings,
                "web.summary.error",
                {"url": url, "title": title, "error": repr(exc)},
            )
            return None

    def _build_search_query(self, query: QueryBundle) -> str:
        candidates = [query.question.strip()]
        candidates.extend(query.keywords[:8])
        cleaned: list[str] = []
        seen: set[str] = set()
        for term in candidates:
            normalized = self._normalize_term(term)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return " ".join(cleaned[:4]) or query.question

    def _normalize_term(self, term: str) -> str:
        term = re.sub(r"\s+", " ", term.strip())
        if not term:
            return ""
        if re.fullmatch(r"[\u4e00-\u9fff]{1,2}", term):
            return ""
        if re.fullmatch(r"[\u4e00-\u9fff]{2}", term):
            return ""
        if re.fullmatch(r"[\u4e00-\u9fff]{3,4}", term):
            return term if self._looks_like_named_entity(term) else ""
        if re.fullmatch(r"[A-Za-z0-9_-]{1,2}", term):
            return ""
        return term

    def _looks_like_named_entity(self, term: str) -> bool:
        suffixes = ("铁塔", "故宫", "大厦", "大桥", "大学", "城市", "建筑", "宫殿", "公园", "博物馆")
        return any(term.endswith(suffix) for suffix in suffixes)
