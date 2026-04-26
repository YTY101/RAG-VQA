from __future__ import annotations

import re

from .schemas import QueryBundle


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "many",
    "much",
    "this",
    "that",
    "there",
    "image",
    "picture",
    "photo",
    "请",
    "问",
    "什么",
    "哪个",
    "哪里",
    "如何",
    "多少",
    "这",
    "这个",
    "图像",
    "图片",
    "照片",
}


class QueryGenerator:
    """Build text and visual queries from the question and image caption."""

    def generate(self, question: str, visual_caption: str) -> QueryBundle:
        keywords = self._keywords(question + " " + visual_caption)
        text_query = " ".join(keywords[:12]) or question
        if visual_caption:
            text_query = f"{text_query} {visual_caption}".strip()
        return QueryBundle(
            question=question,
            visual_caption=visual_caption,
            text_query=text_query,
            keywords=keywords,
        )

    def _keywords(self, text: str) -> list[str]:
        raw_tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z][A-Za-z0-9_-]+|\d+", text.lower())
        tokens: list[str] = []
        for token in raw_tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
                tokens.extend(self._cjk_terms(token))
            else:
                tokens.append(token)
        seen: set[str] = set()
        result: list[str] = []
        for token in tokens:
            if token in STOPWORDS or len(token) <= 1 or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    def _cjk_terms(self, text: str) -> list[str]:
        terms = [text]
        for size in (2, 3, 4):
            terms.extend(text[i : i + size] for i in range(0, max(0, len(text) - size + 1)))
        return terms
