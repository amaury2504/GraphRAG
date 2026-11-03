import pytest

from graphrag.domain.chunking import chunk_abstract, split_sentences


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def sentence(n: int, prefix: str = "w") -> str:
    """A sentence of exactly n words, starting with a capital and ending with a period."""
    return "A" + words(n, prefix)[1:] + "."


def test_short_abstract_is_one_chunk() -> None:
    chunks = chunk_abstract("2005.11401", "We study RAG. It works well.")
    assert len(chunks) == 1
    assert chunks[0].arxiv_id == "2005.11401"
    assert chunks[0].position == 0
    assert chunks[0].text == "We study RAG. It works well."


def test_whitespace_and_newlines_are_collapsed() -> None:
    chunks = chunk_abstract("1", "We study\n  RAG.\n\nIt  works.")
    assert chunks[0].text == "We study RAG. It works."


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_abstract_gives_no_chunk(text: str) -> None:
    assert chunk_abstract("1", text) == ()


def test_exactly_max_words_stays_single() -> None:
    text = sentence(10)
    assert len(chunk_abstract("1", text, max_words=10)) == 1


def test_one_word_over_splits() -> None:
    text = f"{sentence(6, 'a')} {sentence(5, 'b')}"
    chunks = chunk_abstract("1", text, max_words=10)
    assert len(chunks) == 2


def test_split_keeps_sentence_boundaries_and_overlap() -> None:
    s1, s2, s3 = sentence(4, "a"), sentence(4, "b"), sentence(4, "c")
    chunks = chunk_abstract("1", f"{s1} {s2} {s3}", max_words=9)
    assert [c.text for c in chunks] == [f"{s1} {s2}", f"{s2} {s3}"]
    assert [c.position for c in chunks] == [0, 1]


def test_no_overlap_when_it_would_exceed_the_limit() -> None:
    s1, s2 = sentence(8, "a"), sentence(8, "b")
    chunks = chunk_abstract("1", f"{s1} {s2}", max_words=10)
    assert [c.text for c in chunks] == [s1, s2]


def test_no_chunk_exceeds_max_words_and_no_word_is_lost() -> None:
    text = " ".join(sentence(7, f"s{i}_") for i in range(12))
    chunks = chunk_abstract("1", text, max_words=20)
    assert all(len(c.text.split()) <= 20 for c in chunks)
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(text.split())


def test_oversized_sentence_is_cut_by_words() -> None:
    text = sentence(25)
    chunks = chunk_abstract("1", text, max_words=10)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 5]
    assert " ".join(c.text for c in chunks) == text


def test_invalid_max_words() -> None:
    with pytest.raises(ValueError, match="max_words"):
        chunk_abstract("1", "text", max_words=0)


def test_chunking_is_deterministic() -> None:
    text = " ".join(sentence(30, f"s{i}_") for i in range(20))
    assert chunk_abstract("1", text) == chunk_abstract("1", text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two.", ["One.", "Two."]),
        ("Is it? Yes! Fine.", ["Is it?", "Yes!", "Fine."]),
        ("It gains 3.5 points.", ["It gains 3.5 points."]),
        ("Use e.g. the model. Next.", ["Use e.g. the model.", "Next."]),
        ("", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_sentences_summing_exactly_to_max_words_stay_in_one_chunk() -> None:
    s1, s2 = sentence(5, "a"), sentence(5, "b")
    chunks = chunk_abstract("1", f"{s1} {s2}", max_words=10)
    assert [c.text for c in chunks] == [f"{s1} {s2}"]


def test_overlap_is_kept_when_it_fills_the_chunk_exactly() -> None:
    s1, s2, s3 = sentence(5, "a"), sentence(5, "b"), sentence(5, "c")
    chunks = chunk_abstract("1", f"{s1} {s2} {s3}", max_words=10)
    assert [c.text for c in chunks] == [f"{s1} {s2}", f"{s2} {s3}"]
