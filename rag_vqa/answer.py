from __future__ import annotations

import re

from .config import Settings
from .schemas import Evidence, QueryBundle


class AnswerGenerator:
    """Generate final grounded answers from image semantics and evidence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tokenizer = None
        self._model = None
        if settings.enable_generator:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(settings.generator_model)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(settings.generator_model)
                self._model.eval()
            except Exception:
                self._tokenizer = None
                self._model = None

    def generate(self, query: QueryBundle, evidences: list[Evidence], visual_answer: str | None) -> str:
        if self._model is not None and self._tokenizer is not None:
            generated = self._generate_with_model(query, evidences, visual_answer)
            if generated:
                return generated
        return self._extractive_answer(query, evidences, visual_answer)

    def _generate_with_model(self, query: QueryBundle, evidences: list[Evidence], visual_answer: str | None) -> str:
        evidence_text = "\n".join(
            f"[{idx}] {ev.title}: {ev.content}" for idx, ev in enumerate(evidences[:5], start=1)
        )
        prompt = (
            "Answer the question using the image caption, visual answer, and reference evidence. "
            "If evidence is insufficient, say what is uncertain. Keep the answer concise and cite evidence numbers.\n\n"
            f"Image caption: {query.visual_caption}\n"
            f"Visual-only answer: {visual_answer or 'N/A'}\n"
            f"Question: {query.question}\n"
            f"Reference evidence:\n{evidence_text}\n"
            "Final answer:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        output = self._model.generate(**inputs, max_new_tokens=128, num_beams=4)
        text = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return text

    def _extractive_answer(self, query: QueryBundle, evidences: list[Evidence], visual_answer: str | None) -> str:
        if not evidences and visual_answer:
            return f"从图像直接判断，答案可能是：{visual_answer}。当前没有检索到足够外部证据。"
        if not evidences:
            return "未检索到足够证据，无法给出可靠答案。"

        best = evidences[0]
        sentence = self._best_sentence(best.content, query.keywords)
        prefix = ""
        if visual_answer:
            prefix = f"图像直接识别结果提示为“{visual_answer}”。"
        return f"{prefix}结合证据可回答：{sentence}（来源：{best.title}）。"

    def _best_sentence(self, content: str, keywords: list[str]) -> str:
        sentences = re.split(r"(?<=[。！？.!?])\s+", content)
        if not sentences:
            return content[:240]
        scored = []
        for sentence in sentences:
            low = sentence.lower()
            score = sum(1 for k in keywords if k.lower() in low)
            scored.append((score, sentence))
        scored.sort(key=lambda item: (-item[0], len(item[1])))
        return scored[0][1][:260]

