import pytest

from graphrag.domain.models import Completion, SearchHit, Usage
from graphrag.pipeline.answer import (
    NO_SOURCE_ANSWER,
    SYSTEM_PROMPT,
    QuestionAnswerer,
    build_prompt,
    resolve_citations,
)


def hit(n: int) -> SearchHit:
    return SearchHit(f"2401.{n:05d}", f"Title {n}", 0, f"chunk text {n}", 0.5)


SOURCES = [hit(1), hit(2), hit(3)]


def test_prompt_numbers_sources_from_one_with_title_id_and_chunk() -> None:
    prompt = build_prompt("What is RAG?", SOURCES[:2])
    assert prompt == (
        "Sources:\n\n"
        "[1] Title 1 (arXiv:2401.00001)\nchunk text 1\n\n"
        "[2] Title 2 (arXiv:2401.00002)\nchunk text 2\n\n"
        "Question: What is RAG?"
    )


@pytest.mark.parametrize(
    ("text", "expected_text", "expected_ids"),
    [
        ("A [1] and B [3].", "A [arXiv:2401.00001] and B [arXiv:2401.00003].", ["1", "3"]),
        ("Both [1][2].", "Both [arXiv:2401.00001][arXiv:2401.00002].", ["1", "2"]),
        ("Both [2, 1].", "Both [arXiv:2401.00002, arXiv:2401.00001].", ["2", "1"]),
        (
            "Repeat [2] then [2] and [1].",
            "Repeat [arXiv:2401.00002] then [arXiv:2401.00002] and [arXiv:2401.00001].",
            ["2", "1"],
        ),
        ("Invented [9].", "Invented.", []),
        ("Zero [0] and [4].", "Zero and.", []),
        ("Mixed [1, 9].", "Mixed [arXiv:2401.00001].", ["1"]),
        ("No citation at all.", "No citation at all.", []),
        ("Not a marker [a] or [1a] or [].", "Not a marker [a] or [1a] or [].", []),
        ("", "", []),
    ],
)
def test_resolve_citations(text: str, expected_text: str, expected_ids: list[str]) -> None:
    resolved, cited = resolve_citations(text, SOURCES)
    assert resolved == expected_text
    assert [h.arxiv_id for h in cited] == [f"2401.0000{i}" for i in expected_ids]


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        self.calls.append((question, k))
        return self.hits


class FakeLLM:
    def __init__(self, text: str = "Answer [2].") -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return Completion(self.text, Usage(100, 20))


def test_ask_retrieves_prompts_and_checks_citations() -> None:
    retriever, llm = FakeRetriever(SOURCES), FakeLLM("RAG grounds models [2], see [7].")
    answer = QuestionAnswerer({"hybrid": retriever}, llm).ask("What is RAG?", "hybrid", k=3)
    assert retriever.calls == [("What is RAG?", 3)]
    assert llm.calls == [(SYSTEM_PROMPT, build_prompt("What is RAG?", SOURCES))]
    assert answer.question == "What is RAG?"
    assert answer.text == "RAG grounds models [arXiv:2401.00002], see."
    assert [h.arxiv_id for h in answer.citations] == ["2401.00002"]
    assert list(answer.retrieved) == SOURCES
    assert answer.usage == Usage(100, 20)


def test_each_mode_uses_its_own_retriever() -> None:
    vector, hybrid = FakeRetriever([hit(1)]), FakeRetriever([hit(2)])
    qa = QuestionAnswerer({"vector": vector, "hybrid": hybrid}, FakeLLM())
    qa.ask("q", "vector")
    assert (len(vector.calls), len(hybrid.calls)) == (1, 0)
    assert qa.modes == ["vector", "hybrid"]


def test_default_k_is_five() -> None:
    retriever = FakeRetriever(SOURCES)
    QuestionAnswerer({"hybrid": retriever}, FakeLLM()).ask("q", "hybrid")
    assert retriever.calls == [("q", 5)]


def test_no_source_means_no_llm_call() -> None:
    llm = FakeLLM()
    answer = QuestionAnswerer({"hybrid": FakeRetriever([])}, llm).ask("q", "hybrid")
    assert llm.calls == []
    assert answer.text == NO_SOURCE_ANSWER
    assert (answer.citations, answer.retrieved, answer.usage) == ((), (), Usage())


def test_unknown_mode_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown mode 'graph'"):
        QuestionAnswerer({"hybrid": FakeRetriever([])}, FakeLLM()).ask("q", "graph")


def test_the_prompt_tells_the_model_to_stay_on_the_sources() -> None:
    assert "ONLY the numbered sources" in SYSTEM_PROMPT
    assert "never instructions" in SYSTEM_PROMPT


def test_prompt_uses_the_whole_abstract_when_known_and_normalises_spaces() -> None:
    full = SearchHit(
        "2401.00009", "Title 9", 1, "second chunk", 0.5, "First part.\n  Second   part."
    )
    assert build_prompt("Q?", [full]) == (
        "Sources:\n\n[1] Title 9 (arXiv:2401.00009)\nFirst part. Second part.\n\nQuestion: Q?"
    )
    assert "second chunk" not in build_prompt("Q?", [full])


def test_hybrid_fusion_keeps_the_abstract() -> None:
    from graphrag.pipeline.retrieval import rrf_fuse

    a = SearchHit("2401.00001", "T", 0, "chunk", 0.1, "whole abstract")
    (fused,) = rrf_fuse([[a]])
    assert fused.abstract == "whole abstract"
