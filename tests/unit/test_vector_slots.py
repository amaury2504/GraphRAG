"""Where the vectors of an embedding model live: one property and one index per model."""

import re
from dataclasses import replace

import pytest

from graphrag.domain.models import IndexInfo
from graphrag.domain.versioning import LEGACY_SLOT

INFO = IndexInfo("model-a", "rev1", 384, "title. chunk")


def test_the_slot_of_a_model_is_stable_and_made_of_safe_identifiers() -> None:
    slot = INFO.slot()
    assert slot == IndexInfo("model-a", "rev1", 384, "title. chunk").slot()
    assert re.fullmatch(r"embedding_[0-9a-f]{12}", slot.property)
    assert re.fullmatch(r"chunk_embedding_[0-9a-f]{12}", slot.index)
    assert slot.property.removeprefix("embedding_") == slot.index.removeprefix("chunk_embedding_")


@pytest.mark.parametrize(
    "other",
    [
        replace(INFO, model="model-b"),
        replace(INFO, revision="rev2"),
        replace(INFO, dimensions=768),
        replace(INFO, text_template="chunk"),
    ],
)
def test_any_difference_in_how_vectors_are_built_gives_another_slot(other: IndexInfo) -> None:
    assert other.slot() != INFO.slot()


def test_a_slot_never_collides_with_the_legacy_one() -> None:
    assert LEGACY_SLOT.property == "embedding"
    assert LEGACY_SLOT.index == "chunk_embedding"
    assert INFO.slot() != LEGACY_SLOT
