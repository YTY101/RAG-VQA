from __future__ import annotations

from PIL import Image

from rag_vqa.config import Settings
from rag_vqa.query import QueryGenerator
from rag_vqa.retriever import KnowledgeBase
from rag_vqa.schemas import Document
from rag_vqa.answer import AnswerGenerator
from rag_vqa.vision import ImageDescriber, VisualQuestionAnswerer
from rag_vqa.web_retriever import WikipediaRetriever


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


def test_web_retriever_disables_env_proxy_by_default() -> None:
    retriever = WikipediaRetriever()
    assert retriever.session.trust_env is False


def test_web_retriever_can_opt_in_env_proxy() -> None:
    retriever = WikipediaRetriever(use_env_proxy=True)
    assert retriever.session.trust_env is True


def test_web_retriever_builds_cleaner_search_query() -> None:
    retriever = WikipediaRetriever()
    query = QueryGenerator().generate("介绍一下这座建筑", "a photo of the Eiffel Tower in Paris")
    search_query = retriever._build_search_query(query)
    terms = search_query.split()
    assert "介绍一下这座建筑" in search_query
    assert "介绍" not in terms
    assert "一下" not in terms


def test_web_retriever_uses_query_search_api(monkeypatch) -> None:
    retriever = WikipediaRetriever()
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"query": {"search": [{"title": "埃菲尔铁塔"}]}}

    def fake_get(url, params=None, timeout=None, headers=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        captured["headers"] = headers
        return DummyResponse()

    monkeypatch.setattr(retriever.session, "get", fake_get)
    titles = retriever._search_titles("埃菲尔铁塔 历史意义", top_k=2)

    assert titles == ["埃菲尔铁塔"]
    assert captured["params"] == {
        "action": "query",
        "list": "search",
        "srsearch": "埃菲尔铁塔 历史意义",
        "srlimit": 2,
        "format": "json",
    }


def test_image_describer_fallback_uses_filename_and_color(tmp_path) -> None:
    image = tmp_path / "blue_tower.png"
    Image.new("RGB", (32, 32), color=(20, 40, 220)).save(image)
    describer = ImageDescriber("missing-model")
    describer._processor = None
    describer._model = None
    text = describer.describe(image)
    assert "blue tower" in text.lower()
    assert "blue" in text.lower()


def test_visual_question_answerer_returns_none_when_disabled() -> None:
    answerer = VisualQuestionAnswerer("missing-model", enabled=False)
    assert answerer.answer("/tmp/not-used.png", "what is this?") is None


def test_answer_generator_stays_disabled_when_generation_off() -> None:
    generator = AnswerGenerator(Settings(enable_generator=False))
    assert generator._model is None
    assert generator._tokenizer is None
    assert generator._device == "cpu"
