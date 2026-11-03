import pytest

from graphrag.domain.identifiers import (
    arxiv_doi_url,
    arxiv_id_from_doi,
    normalize_arxiv_id,
    short_openalex_id,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2005.11401", "2005.11401"),
        ("2005.11401v3", "2005.11401"),
        ("http://arxiv.org/abs/2005.11401v1", "2005.11401"),
        ("https://arxiv.org/abs/2312.10997", "2312.10997"),
        ("arXiv:2401.0123", "2401.0123"),
        ("  2005.11401  ", "2005.11401"),
        ("cs/0112017v2", "cs/0112017"),
        ("hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_normalize_arxiv_id(raw: str, expected: str) -> None:
    assert normalize_arxiv_id(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "abc", "2005.114", "2005.11401v", "http://example.com/2005.11401"]
)
def test_normalize_rejects_garbage(raw: str) -> None:
    with pytest.raises(ValueError, match="not an arXiv identifier"):
        normalize_arxiv_id(raw)


def test_doi_roundtrip() -> None:
    url = arxiv_doi_url("2005.11401v2")
    assert url == "https://doi.org/10.48550/arXiv.2005.11401"
    assert arxiv_id_from_doi(url) == "2005.11401"


def test_doi_from_openalex_is_lowercase() -> None:
    assert arxiv_id_from_doi("https://doi.org/10.48550/arxiv.2312.10997") == "2312.10997"


@pytest.mark.parametrize(
    "doi", ["https://doi.org/10.1000/xyz", "", "https://doi.org/10.48550/arxiv.nope"]
)
def test_non_arxiv_doi_is_none(doi: str) -> None:
    assert arxiv_id_from_doi(doi) is None


def test_short_openalex_id() -> None:
    assert short_openalex_id("https://openalex.org/W3027879771") == "W3027879771"
    assert short_openalex_id("W3027879771") == "W3027879771"
