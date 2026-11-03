import json
from pathlib import Path

import httpx
import pytest

from graphrag.adapters.openalex import BATCH_SIZE, OpenAlexSource, parse_works

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "openalex_page.json"


def test_parse_real_payload() -> None:
    records = {r.arxiv_id: r for r in parse_works(json.loads(FIXTURE.read_text("utf-8")))}
    assert set(records) == {"2005.11401", "2312.10997"}
    lewis = records["2005.11401"]
    assert lewis.openalex_id == "W3027879771"
    assert lewis.authors[0].key == "A5063377058"
    assert lewis.authors[0].name == "Patrick Lewis"
    assert len(lewis.authors) == 12
    assert len(lewis.referenced_openalex_ids) == 59
    assert all(r.startswith("W") for r in lewis.referenced_openalex_ids)
    # Known coverage gap: this survey has no references in OpenAlex.
    assert records["2312.10997"].referenced_openalex_ids == ()


def test_parse_skips_non_arxiv_and_authors_without_id() -> None:
    payload = {
        "results": [
            {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1000/x"},
            {"id": "https://openalex.org/W2", "doi": None},
            {
                "id": "https://openalex.org/W3",
                "doi": "https://doi.org/10.48550/arxiv.2401.00001",
                "authorships": [
                    {"author": {"id": "https://openalex.org/A1", "display_name": "Ann"}},
                    {"author": {"id": None, "display_name": "Ghost"}},
                    {"author": None},
                ],
                "referenced_works": None,
            },
        ]
    }
    records = parse_works(payload)
    assert [r.arxiv_id for r in records] == ["2401.00001"]
    assert [a.key for a in records[0].authors] == ["A1"]
    assert records[0].referenced_openalex_ids == ()


def test_parse_deduplicates_same_arxiv_id() -> None:
    work = {"id": "https://openalex.org/W3", "doi": "https://doi.org/10.48550/arxiv.2401.00001"}
    assert len(parse_works({"results": [work, work]})) == 1


def test_parse_rejects_unexpected_payload() -> None:
    with pytest.raises(ValueError, match="no results"):
        parse_works({"error": "nope"})


def test_fetch_builds_one_request_with_or_filter() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=json.loads(FIXTURE.read_text("utf-8")))

    source = OpenAlexSource(httpx.Client(transport=httpx.MockTransport(handler)))
    records = source.fetch_works(["2005.11401", "2312.10997v2"])
    assert len(records) == 2
    assert len(seen) == 1
    flt = seen[0].url.params["filter"]
    assert flt == (
        "doi:https://doi.org/10.48550/arXiv.2005.11401|https://doi.org/10.48550/arXiv.2312.10997"
    )
    assert seen[0].url.params["per_page"] == "100"


def test_fetch_empty_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    source = OpenAlexSource(httpx.Client(transport=httpx.MockTransport(handler)))
    assert source.fetch_works([]) == []


def test_fetch_rejects_oversized_batch() -> None:
    source = OpenAlexSource(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    )
    with pytest.raises(ValueError, match="at most"):
        source.fetch_works([f"2401.{i:05d}" for i in range(BATCH_SIZE + 1)])
