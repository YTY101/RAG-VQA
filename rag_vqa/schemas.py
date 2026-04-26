from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """One searchable knowledge item."""

    id: str
    title: str
    text: str
    source: str
    type: str = "text"
    image_path: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryBundle:
    question: str
    visual_caption: str
    text_query: str
    keywords: list[str]


@dataclass
class Evidence:
    id: str
    title: str
    content: str
    source: str
    type: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAnswer:
    answer: str
    visual_caption: str
    visual_answer: str | None
    query: QueryBundle
    evidences: list[Evidence]

