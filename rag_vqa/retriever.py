from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .debug import debug_dump
from .embeddings import ImageEmbedder, TextEmbedder, l2_normalize
from .schemas import Document, Evidence, QueryBundle


class KnowledgeBase:
    """Local vector store for text evidence and optional image evidence."""

    def __init__(
        self,
        settings: Settings,
        docs: list[Document] | None = None,
        text_vectors: np.ndarray | None = None,
        image_vectors: np.ndarray | None = None,
    ) -> None:
        self.settings = settings
        self.docs = docs or []
        self.text_embedder = TextEmbedder(settings.text_embedding_model)
        self.image_embedder = ImageEmbedder(settings.image_embedding_model)
        self.text_vectors = text_vectors
        self.image_vectors = image_vectors

    @classmethod
    def from_jsonl(cls, path: str | Path, settings: Settings) -> "KnowledgeBase":
        docs: list[Document] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                docs.append(Document(**item))
        kb = cls(settings=settings, docs=docs)
        kb.build()
        return kb

    @classmethod
    def load(cls, index_dir: str | Path, settings: Settings) -> "KnowledgeBase":
        index_dir = Path(index_dir)
        with (index_dir / "documents.json").open("r", encoding="utf-8") as f:
            docs = [Document(**item) for item in json.load(f)]
        text_vectors = np.load(index_dir / "text_vectors.npy")
        image_vectors = np.load(index_dir / "image_vectors.npy")
        return cls(settings=settings, docs=docs, text_vectors=text_vectors, image_vectors=image_vectors)

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "documents.json").open("w", encoding="utf-8") as f:
            json.dump([doc.__dict__ for doc in self.docs], f, ensure_ascii=False, indent=2)
        np.save(index_dir / "text_vectors.npy", self.text_vectors)
        np.save(index_dir / "image_vectors.npy", self.image_vectors)

    def build(self) -> None:
        texts = [self._doc_text(doc) for doc in self.docs]
        self.text_vectors = self.text_embedder.encode(texts)
        image_vectors = []
        for doc in self.docs:
            if doc.image_path and Path(doc.image_path).exists():
                image_vectors.append(self.image_embedder.encode_paths([doc.image_path])[0])
            else:
                image_vectors.append(np.zeros(self.image_embedder.dim, dtype=np.float32))
        self.image_vectors = np.vstack(image_vectors).astype(np.float32) if image_vectors else np.zeros((0, 96), dtype=np.float32)
        debug_dump(
            self.settings,
            "index.build",
            {
                "doc_count": len(self.docs),
                "text_vector_shape": self.text_vectors.shape,
                "image_vector_shape": self.image_vectors.shape,
                "text_embedding_model": self.text_embedder.model_name,
                "image_embedding_model": self.image_embedder.model_name,
            },
        )

    def retrieve(self, query: QueryBundle, image_path: str | Path, top_k: int) -> list[Evidence]:
        if self.text_vectors is None or self.image_vectors is None:
            self.build()

        text_scores = self._text_scores(query)
        image_scores = self._image_scores(image_path)
        text_scores = self._safe_align(text_scores, len(self.docs))
        image_scores = self._safe_align(image_scores, len(self.docs))

        combined = self.settings.text_weight * text_scores + self.settings.image_weight * image_scores
        ranked_idx = np.argsort(-combined)
        debug_dump(
            self.settings,
            "step2.retrieval_scores",
            {
                "text_weight": self.settings.text_weight,
                "image_weight": self.settings.image_weight,
                "min_evidence_score": self.settings.min_evidence_score,
                "scores": [
                    {
                        "rank": rank + 1,
                        "doc_id": self.docs[int(idx)].id,
                        "title": self.docs[int(idx)].title,
                        "text_score": float(text_scores[int(idx)]),
                        "image_score": float(image_scores[int(idx)]),
                        "combined_score": float(combined[int(idx)]),
                    }
                    for rank, idx in enumerate(ranked_idx[: max(top_k, 10)])
                ],
            },
        )
        evidences: list[Evidence] = []
        seen: set[str] = set()
        for idx in ranked_idx:
            if len(evidences) >= top_k:
                break
            score = float(combined[idx])
            if score < self.settings.min_evidence_score:
                continue
            doc = self.docs[int(idx)]
            fingerprint = self._fingerprint(doc.text)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            evidences.append(
                Evidence(
                    id=doc.id,
                    title=doc.title,
                    content=self._snippet(doc.text, query.keywords),
                    source=doc.source,
                    type=doc.type,
                    score=score,
                    metadata={**doc.metadata, "image_path": doc.image_path, "tags": doc.tags},
                )
            )
        return evidences

    def _text_scores(self, query: QueryBundle) -> np.ndarray:
        assert self.text_vectors is not None
        q = self.text_embedder.encode([query.text_query])[0]
        return np.dot(self.text_vectors, q)

    def _image_scores(self, image_path: str | Path) -> np.ndarray:
        assert self.image_vectors is not None
        if not Path(image_path).exists() or not self.image_vectors.any():
            return np.zeros(len(self.docs), dtype=np.float32)
        try:
            Image.open(image_path).verify()
            q = self.image_embedder.encode_paths([image_path])[0]
            return np.dot(l2_normalize(self.image_vectors), q)
        except Exception:
            return np.zeros(len(self.docs), dtype=np.float32)

    def _safe_align(self, scores: np.ndarray, n: int) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape[0] == n:
            return scores
        fixed = np.zeros(n, dtype=np.float32)
        fixed[: min(n, scores.shape[0])] = scores[: min(n, scores.shape[0])]
        return fixed

    def _doc_text(self, doc: Document) -> str:
        tags = " ".join(doc.tags)
        return f"{doc.title}\n{doc.text}\n{tags}"

    def _snippet(self, text: str, keywords: list[str], max_chars: int = 450) -> str:
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text
        lower = text.lower()
        positions = [lower.find(k.lower()) for k in keywords if lower.find(k.lower()) >= 0]
        start = max(0, min(positions) - 80) if positions else 0
        end = min(len(text), start + max_chars)
        return text[start:end].strip()

    def _fingerprint(self, text: str) -> str:
        normalized = "".join(text.lower().split())[:220]
        return normalized
