from __future__ import annotations

from difflib import SequenceMatcher
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
        # language: str = "zh",
        language: str = "en",
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

    def _get_json(self, url: str, *, params: dict | None = None, headers: dict | None = None) -> dict | list:
        request_headers = headers or {"accept": "application/json"}
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout, headers=request_headers)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            resp = requests.get(
                url,
                params=params,
                timeout=self.timeout,
                headers={**self.session.headers, **request_headers},
            )
            resp.raise_for_status()
            return resp.json()

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
        titles = self._search_titles(search_query, top_k=top_k, query=query)
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
                    score=self._evidence_score(query, title, summary),
                )
            )
        return sorted(evidences, key=lambda item: item.score, reverse=True)[:top_k]

    def _search_titles(self, search_query: str, top_k: int, query: QueryBundle) -> list[str]:
        primary_results = self._wiki_search(search_query, limit=max(top_k * 3, 5))
        titles = [item["title"] for item in primary_results]

        best_primary = self._best_result_score(query, primary_results)
        suggestion = self._best_suggestion(search_query, query.question)
        fallback_queries = self._fallback_queries(query, suggestion)

        if best_primary < 0.55:
            for fallback_query in fallback_queries:
                fallback_results = self._wiki_search(fallback_query, limit=max(top_k * 3, 5))
                titles.extend(item["title"] for item in fallback_results)
                if self._ranked_title_score(query, titles) >= 0.7 and len({title.lower() for title in titles}) >= top_k:
                    break

        ranked_titles = self._rank_titles(query, titles)
        debug_dump(
            self.settings,
            "web.search.results",
            {
                "search_query": search_query,
                "suggestion": suggestion,
                "best_primary": best_primary,
                "fallback_queries": fallback_queries,
                "titles": ranked_titles,
            },
        )
        return ranked_titles[:top_k]

    def _wiki_search(self, search_query: str, limit: int) -> list[dict]:
        url = f"https://{self.language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "srlimit": limit,
            "srinfo": "suggestion",
            "format": "json",
        }
        try:
            data = self._get_json(url, params=params)
            return [
                {
                    "title": str(item.get("title", "")).strip(),
                    "snippet": self._strip_html(str(item.get("snippet", "")).strip()),
                }
                for item in data.get("query", {}).get("search", [])
                if str(item.get("title", "")).strip()
            ]
        except Exception as exc:
            debug_dump(
                self.settings,
                "web.search.error",
                {"url": url, "search_query": search_query, "error": repr(exc)},
            )
            return []

    def _search_suggestion(self, search_query: str) -> str | None:
        url = f"https://{self.language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "srlimit": 1,
            "srinfo": "suggestion",
            "format": "json",
        }
        try:
            data = self._get_json(url, params=params)
            suggestion = data.get("query", {}).get("searchinfo", {}).get("suggestion")
            if isinstance(suggestion, str) and suggestion.strip():
                return suggestion.strip()
            return None
        except Exception:
            return None

    def _best_suggestion(self, *queries: str) -> str | None:
        for candidate in queries:
            suggestion = self._search_suggestion(candidate)
            if suggestion:
                return suggestion
            suggestion = self._opensearch_suggestion(candidate)
            if suggestion:
                return suggestion
        return None

    def _opensearch_suggestion(self, search_query: str) -> str | None:
        url = f"https://{self.language}.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": search_query,
            "limit": 3,
            "namespace": 0,
            "format": "json",
        }
        try:
            data = self._get_json(url, params=params)
            suggestions = data[1] if isinstance(data, list) and len(data) > 1 else []
            for item in suggestions:
                if isinstance(item, str) and item.strip():
                    return item.strip()
            return None
        except Exception:
            return None

    def _summary(self, title: str) -> str | None:
        url = f"https://{self.language}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        try:
            data = self._get_json(url, headers={"accept": "application/json"})
            return data.get("extract")
        except Exception as exc:
            debug_dump(
                self.settings,
                "web.summary.error",
                {"url": url, "title": title, "error": repr(exc)},
            )
            return None

    def _build_search_query(self, query: QueryBundle) -> str:
        candidates: list[str] = []
        seen: set[str] = set()
        for term in query.keywords:
            normalized = self._normalize_term(term)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(normalized)
        return " ".join(candidates[:6]) or query.question.strip()

    def _normalize_term(self, term: str) -> str:
        term = re.sub(r"\s+", " ", term.strip())
        if not term:
            return ""
        if term.lower() in {
            "introduce",
            "introduction",
            "introduced",
            "building",
            "buildings",
            "about",
            "tell",
            "describe",
            "description",
            "information",
            "info",
            "shown",
            "show",
            "image",
            "photo",
            "picture",
            "parisian",
            "famous",
            "famouss",
        }:
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

    def _fallback_queries(self, query: QueryBundle, suggestion: str | None) -> list[str]:
        cleaned_terms = self._extract_search_terms(suggestion or "")
        if not cleaned_terms:
            cleaned_terms = self._extract_search_terms(query.question)
        queries: list[str] = []
        if cleaned_terms:
            queries.append(" ".join(cleaned_terms[:4]))
            wildcard_queries = self._wildcard_queries(cleaned_terms)
            queries.extend(wildcard_queries)
            for size in range(min(3, len(cleaned_terms)), 1, -1):
                for idx in range(0, len(cleaned_terms) - size + 1):
                    phrase_terms = cleaned_terms[idx : idx + size]
                    if any(term in {"paris", "beijing", "china", "france"} for term in phrase_terms) and size > 2:
                        continue
                    phrase = " ".join(phrase_terms)
                    queries.append(f'intitle:"{phrase}"')
        deduped: list[str] = []
        seen: set[str] = set()
        for item in queries:
            normalized = item.lower().strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(item)
        return deduped[:8]

    def _extract_search_terms(self, text: str) -> list[str]:
        tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z][A-Za-z0-9_-]+|\d+", text.lower())
        terms: list[str] = []
        for token in tokens:
            normalized = self._normalize_term(token)
            if normalized:
                terms.append(normalized.lower())
        return terms

    def _best_result_score(self, query: QueryBundle, results: list[dict]) -> float:
        if not results:
            return 0.0
        return max(self._candidate_score(query, item["title"], item.get("snippet", "")) for item in results)

    def _rank_titles(self, query: QueryBundle, titles: list[str]) -> list[str]:
        seen: set[str] = set()
        unique_titles: list[str] = []
        for title in titles:
            key = title.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique_titles.append(title)
        base_counts: dict[str, int] = {}
        for title in unique_titles:
            base = self._base_title(title)
            base_counts[base] = base_counts.get(base, 0) + 1
        return sorted(
            unique_titles,
            key=lambda title: self._candidate_score(query, title) + self._canonical_title_bonus(title, base_counts),
            reverse=True,
        )

    def _evidence_score(self, query: QueryBundle, title: str, summary: str) -> float:
        return max(0.2, min(0.95, self._candidate_score(query, title, summary)))

    def _ranked_title_score(self, query: QueryBundle, titles: list[str]) -> float:
        if not titles:
            return 0.0
        ranked = self._rank_titles(query, titles)
        if not ranked:
            return 0.0
        return self._candidate_score(query, ranked[0])

    def _candidate_score(self, query: QueryBundle, title: str, extra_text: str = "") -> float:
        query_terms = set(self._extract_search_terms(f"{query.question} {query.visual_caption} {' '.join(query.keywords)}"))
        text = f"{title} {extra_text}".lower()
        title_terms = set(self._extract_search_terms(text))
        if not query_terms or not title_terms:
            return 0.0
        overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
        fuzzy = SequenceMatcher(None, " ".join(sorted(query_terms)), " ".join(sorted(title_terms))).ratio()
        title_bonus = 0.15 if any(term in title.lower() for term in query_terms if len(term) > 3) else 0.0
        primary_title_bonus = self._primary_title_bonus(title, query_terms)
        disambiguation_penalty = self._disambiguation_penalty(title, query_terms)
        return overlap * 0.7 + fuzzy * 0.2 + title_bonus + primary_title_bonus - disambiguation_penalty

    def _strip_html(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text).strip()

    def _wildcard_queries(self, terms: list[str]) -> list[str]:
        queries: list[str] = []
        if len(terms) < 2:
            return queries
        tail = " ".join(terms[1:4])
        first = terms[0]
        if len(first) >= 4:
            queries.append(f"{first}* {tail}".strip())
            queries.append(f"intitle:{first}* {tail}".strip())
        for idx, term in enumerate(terms[:-1]):
            if len(term) < 4:
                continue
            remainder = " ".join(terms[idx + 1 : idx + 4])
            if remainder:
                queries.append(f"{term}* {remainder}".strip())
                queries.append(f"intitle:{term}* {remainder}".strip())
        return queries

    def _disambiguation_penalty(self, title: str, query_terms: set[str]) -> float:
        match = re.search(r"\(([^)]+)\)", title)
        if not match:
            return 0.0
        qualifier_terms = set(self._extract_search_terms(match.group(1)))
        if not qualifier_terms:
            return 0.05
        if qualifier_terms & query_terms:
            return 0.05
        return 0.18

    def _primary_title_bonus(self, title: str, query_terms: set[str]) -> float:
        if "(" in title and ")" in title:
            return 0.0
        title_terms = set(self._extract_search_terms(title))
        overlap_terms = query_terms & title_terms
        if len(overlap_terms) >= 2:
            return 0.18
        return 0.0

    def _base_title(self, title: str) -> str:
        return re.sub(r"\s*\(.*\)\s*$", "", title).strip().lower()

    def _canonical_title_bonus(self, title: str, base_counts: dict[str, int]) -> float:
        base = self._base_title(title)
        if title.strip().lower() == base and base_counts.get(base, 0) > 1:
            return 0.25
        if "(" in title and ")" in title and base_counts.get(base, 0) > 1:
            return -0.12
        return 0.0
