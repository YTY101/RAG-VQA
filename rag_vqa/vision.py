from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageStat


class ImageDescriber:
    """Generate a short visual description for query construction."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None
        try:
            from transformers import pipeline

            self._pipeline = pipeline("image-to-text", model=model_name)
        except Exception:
            self._pipeline = None

    def describe(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if self._pipeline is not None:
            result = self._pipeline(str(path))
            if result and isinstance(result, list):
                return str(result[0].get("generated_text", "")).strip()
        return self._fallback_description(path)

    def _fallback_description(self, path: Path) -> str:
        image = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(image.resize((64, 64)))
        avg = stat.mean
        dominant = max(range(3), key=lambda i: avg[i])
        color = ["red", "green", "blue"][dominant]
        stem = path.stem.replace("_", " ").replace("-", " ")
        return f"Image named '{stem}', with a visually dominant {color} color tone."


class VisualQuestionAnswerer:
    """Direct VQA model used as visual-only evidence."""

    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._pipeline = None
        if enabled:
            try:
                from transformers import pipeline

                self._pipeline = pipeline("visual-question-answering", model=model_name)
            except Exception:
                self._pipeline = None

    def answer(self, image_path: str | Path, question: str) -> str | None:
        if self._pipeline is None:
            return None
        result = self._pipeline(str(image_path), question)
        if isinstance(result, list) and result:
            best = max(result, key=lambda item: float(item.get("score", 0.0)))
            return str(best.get("answer", "")).strip() or None
        if isinstance(result, dict):
            return str(result.get("answer", "")).strip() or None
        return None

