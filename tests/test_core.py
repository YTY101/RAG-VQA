from __future__ import annotations

from PIL import Image

from rag_vqa.config import Settings
from rag_vqa.query import QueryGenerator
from rag_vqa.retriever import KnowledgeBase
from rag_vqa.schemas import Document


def test_query_generator_uses_question_and_caption() -> None:
    query = QueryGenerator().generate("这座建筑有什么历史意义？", "a photo of the Eiffel Tower in Paris")
    assert "eiffel" in query.text_query.lower()
    assert query.question.startswith("这座建筑")


def test_local_retrieval_returns_relevant_text(tmp_path) -> None:
    image = tmp_path / "red.jpg"
    Image.new("RGB", (32, 32), color=(220, 20, 20)).save(image)
    settings = Settings(enable_generator=False, enable_blip_vqa=False, min_evidence_score=-1.0)
    docs = [
        Document(
            id="fire",
            title="灭火器",
            text="灭火器是一种红色消防设备，常见于室内公共空间。",
            source="test",
            tags=["消防", "红色"],
        ),
        Document(
            id="phone",
            title="手机",
            text="手机是一种电子通信设备。",
            source="test",
            tags=["电子"],
        ),
    ]
    kb = KnowledgeBase(settings=settings, docs=docs)
    kb.build()
    query = QueryGenerator().generate("图中的红色消防设备是什么？", "red object indoors")
    evidences = kb.retrieve(query, image, top_k=1)
    assert evidences
    assert evidences[0].id == "fire"

