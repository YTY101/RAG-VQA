from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        norm = np.linalg.norm(x) + 1e-12
        return x / norm
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


class TextEmbedder:
    """SentenceTransformer embedding with a deterministic hashing fallback."""

    def __init__(self, model_name: str, dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype=np.float32)
        return np.vstack([self._hash_embed(text) for text in texts]).astype(np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[idx] += sign
        return l2_normalize(vector)


class ImageEmbedder:
    """CLIP image embedding with a color-histogram fallback."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.dim = 96
        self._model = None
        self._processor = None
        self._torch = None
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(model_name)
            self._model.eval()
            self.dim = int(self._model.config.projection_dim)
        except Exception:
            self._model = None
            self._processor = None
            self._torch = None

    def encode_paths(self, image_paths: list[str | Path]) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        images = [Image.open(path).convert("RGB") for path in image_paths]
        return self.encode_images(images)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None and self._processor is not None and self._torch is not None:
            with self._torch.no_grad():
                inputs = self._processor(images=images, return_tensors="pt")
                features = self._model.get_image_features(**inputs)
            return l2_normalize(features.cpu().numpy())
        return np.vstack([self._histogram_embed(image) for image in images]).astype(np.float32)

    def _histogram_embed(self, image: Image.Image) -> np.ndarray:
        image = image.resize((224, 224)).convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
        parts = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=32, range=(0, 256))
            parts.append(hist.astype(np.float32))
        return l2_normalize(np.concatenate(parts))

