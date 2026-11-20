import pytest

from graphrag.adapters.sentence_transformer import (
    MODEL_NAME,
    MODEL_REVISION,
    QUERY_INSTRUCTION,
    EmbeddingModelSpec,
    index_info_for,
    load_embedder,
    model_spec,
)
from graphrag.domain.chunking import TEXT_TEMPLATE, search_text
from graphrag.domain.models import IndexInfo

BASE = IndexInfo("bge", "abc", 384, "title. chunk")


def test_identical_fingerprints_have_no_difference() -> None:
    assert BASE.differences(IndexInfo("bge", "abc", 384, "title. chunk")) == []


def test_a_model_change_is_reported_even_with_the_same_dimensions() -> None:
    other = IndexInfo("minilm", "abc", 384, "title. chunk")
    assert BASE.differences(other) == ["model: stored 'bge', current 'minilm'"]


def test_every_field_is_compared() -> None:
    other = IndexInfo("m2", "r2", 768, "chunk")
    assert [d.split(":")[0] for d in BASE.differences(other)] == [
        "model",
        "revision",
        "dimensions",
        "text_template",
    ]


def test_default_info_describes_the_pinned_model_and_the_title_plus_chunk_text() -> None:
    info = index_info_for(MODEL_NAME, 384)
    assert info == IndexInfo(MODEL_NAME, MODEL_REVISION, 384, TEXT_TEMPLATE)
    assert TEXT_TEMPLATE == "title. chunk"
    assert search_text("Title", "Chunk") == "Title. Chunk"


OTHER = {"other/model": EmbeddingModelSpec("other/model", "rev9", "Query: ")}


def test_a_model_of_the_registry_gives_its_own_pinned_revision() -> None:
    assert index_info_for("other/model", 768, OTHER) == IndexInfo(
        "other/model", "rev9", 768, TEXT_TEMPLATE
    )
    assert model_spec(MODEL_NAME).query_instruction == QUERY_INSTRUCTION


def test_an_unknown_model_is_refused_and_the_known_ones_are_listed() -> None:
    with pytest.raises(ValueError, match="unknown embedding model 'nope'; known models: "):
        index_info_for("nope", 3)
    with pytest.raises(ValueError, match="known models: other/model"):
        load_embedder("nope", registry=OTHER)


def test_loading_pins_the_revision_of_the_chosen_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: dict[str, object] = {}

    class FakeModel:
        def __init__(self, name: str, **kwargs: object) -> None:
            loaded.update(name=name, **kwargs)

        def get_embedding_dimension(self) -> int:
            return 3

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    embedder = load_embedder("other/model", cache_folder="cache", registry=OTHER)
    assert loaded == {
        "name": "other/model",
        "revision": "rev9",
        "device": "cpu",
        "cache_folder": "cache",
    }
    assert embedder.dimensions == 3
