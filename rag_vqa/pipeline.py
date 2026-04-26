from __future__ import annotations

from pathlib import Path

from .answer import AnswerGenerator
from .config import Settings
from .debug import debug_dump
from .query import QueryGenerator
from .retriever import KnowledgeBase
from .schemas import Evidence, RAGAnswer
from .vision import ImageDescriber, VisualQuestionAnswerer
from .web_retriever import WikipediaRetriever


class RAGVQAPipeline:
    """End-to-end RAG visual question answering pipeline."""

    def __init__(self, kb: KnowledgeBase, settings: Settings | None = None, enable_web: bool = False) -> None:
        self.settings = settings or Settings()
        self.kb = kb
        self.describer = ImageDescriber(self.settings.caption_model, settings=self.settings)
        self.vqa = VisualQuestionAnswerer(self.settings.vqa_model, settings=self.settings, enabled=self.settings.enable_blip_vqa)
        self.query_generator = QueryGenerator()
        self.answer_generator = AnswerGenerator(self.settings)
        self.web = (
            WikipediaRetriever(
                timeout=self.settings.web_timeout,
                settings=self.settings,
                use_env_proxy=self.settings.web_use_env_proxy,
            )
            if enable_web
            else None
        )

    def ask(self, image_path: str | Path, question: str, top_k: int | None = None) -> RAGAnswer:
        top_k = top_k or self.settings.top_k
        debug_dump(
            self.settings,
            "pipeline.start",
            {
                "image_path": str(image_path),
                "question": question,
                "top_k": top_k,
                "enable_web": self.web is not None,
                "knowledge_base_docs": len(self.kb.docs),
            },
        )

        visual_caption = self.describer.describe(image_path)
        debug_dump(self.settings, "step1.visual_caption", {"visual_caption": visual_caption})

        query = self.query_generator.generate(question, visual_caption)
        debug_dump(self.settings, "step1.query_bundle", query)

        visual_answer = self.vqa.answer(image_path, question)
        debug_dump(self.settings, "step1.visual_answer", {"visual_answer": visual_answer})

        local_evidence = self.kb.retrieve(query, image_path, top_k=top_k)
        debug_dump(self.settings, "step2.local_evidence", local_evidence)

        web_evidence = self.web.retrieve(query, top_k=max(1, top_k // 2)) if self.web else []
        debug_dump(self.settings, "step2.web_evidence", web_evidence)

        evidences = self._merge_evidence(local_evidence + web_evidence, top_k=top_k)
        debug_dump(self.settings, "step3.merged_evidence", evidences)

        answer = self.answer_generator.generate(query, evidences, visual_answer)
        debug_dump(self.settings, "step4.final_answer", {"answer": answer})

        return RAGAnswer(
            answer=answer,
            visual_caption=visual_caption,
            visual_answer=visual_answer,
            query=query,
            evidences=evidences,
        )

    def _merge_evidence(self, evidences: list[Evidence], top_k: int) -> list[Evidence]:
        seen: set[str] = set()
        merged: list[Evidence] = []
        for ev in sorted(evidences, key=lambda item: item.score, reverse=True):
            key = (ev.title + ev.content[:80]).lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(ev)
            if len(merged) >= top_k:
                break
        return merged
