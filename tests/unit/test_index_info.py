from graphrag.adapters.sentence_transformer import MODEL_NAME, MODEL_REVISION, default_index_info
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
    info = default_index_info(384)
    assert info == IndexInfo(MODEL_NAME, MODEL_REVISION, 384, TEXT_TEMPLATE)
    assert TEXT_TEMPLATE == "title. chunk"
    assert search_text("Title", "Chunk") == "Title. Chunk"
