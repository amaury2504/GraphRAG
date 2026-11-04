import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from graphrag.adapters.http import RequestFailedError
from graphrag.adapters.semantic_scholar import BATCH_SIZE, SemanticScholarSource, parse_batch
from graphrag.domain.models import ReferenceRecord

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "semantic_scholar_batch.json"
IDS = ["2401.00001", "9999.99999", "2401.00003"]


def sample_payload() -> list[Any]:
    payload: list[Any] = json.loads(FIXTURE.read_text("utf-8"))
    return payload


def test_parse_sample_payload() -> None:
    records = parse_batch(IDS, sample_payload())
    assert [r.arxiv_id for r in records] == ["2401.00001", "2401.00003"]  # unknown id skipped
    # references without an arXiv id (ACL-only here) are dropped
    assert records[0].cited_arxiv_ids == ("2312.00002", "2401.00003")
    assert records[1].cited_arxiv_ids == ()


def test_unknown_papers_are_absent_not_empty() -> None:
    assert parse_batch(["2401.00001"], [None]) == []


def test_reference_without_arxiv_id_gives_empty_record() -> None:
    payload = [
        {"externalIds": {"ArXiv": "2401.00001"}, "references": [{"externalIds": {"DOI": "x"}}]}
    ]
    assert parse_batch(["2401.00001"], payload) == [ReferenceRecord("2401.00001", ())]


def test_robust_to_null_fields_and_duplicates_and_self_reference() -> None:
    payload = [
        {
            "externalIds": {"ArXiv": "2401.00001"},
            "references": [
                None,
                {"externalIds": None},
                {"externalIds": {"ArXiv": "2401.00002"}},
                {"externalIds": {"ArXiv": "2401.00002v2"}},  # same paper, versioned
                {"externalIds": {"ArXiv": "2401.00001"}},  # self citation
                {"externalIds": {"ArXiv": "not-an-id"}},
            ],
        }
    ]
    assert parse_batch(["2401.00001"], payload) == [ReferenceRecord("2401.00001", ("2401.00002",))]


def test_references_missing_entirely() -> None:
    payload = [{"externalIds": {"ArXiv": "2401.00001"}, "references": None}]
    assert parse_batch(["2401.00001"], payload) == [ReferenceRecord("2401.00001", ())]


def test_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="asked for 2"):
        parse_batch(["2401.00001", "2401.00002"], [None])


def test_order_mismatch_is_an_error() -> None:
    # the first result carries the id of the second requested paper: the order is broken
    payload = [
        {"externalIds": {"ArXiv": "2401.00002"}, "references": []},
        {"externalIds": {"ArXiv": "2401.00001"}, "references": []},
    ]
    with pytest.raises(ValueError, match=r"order mismatch: asked 2401\.00001, got 2401\.00002"):
        parse_batch(["2401.00001", "2401.00002"], payload)


def test_versioned_id_of_another_batch_paper_is_still_a_mismatch() -> None:
    payload = [
        {"externalIds": {"ArXiv": "2401.00002v3"}, "references": []},
        {"externalIds": {"ArXiv": "2401.00001"}, "references": []},
    ]
    with pytest.raises(ValueError, match="order mismatch"):
        parse_batch(["2401.00001", "2401.00002"], payload)


def test_alias_outside_the_batch_keeps_references_under_requested_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # same work deposited twice on arXiv: Semantic Scholar returns the other id
    payload = [
        {
            "externalIds": {"ArXiv": "2310.00099"},
            "references": [
                {"externalIds": {"ArXiv": "2401.00005"}},
                {"externalIds": {"ArXiv": "2310.00099v2"}},  # its own alias, not a citation
            ],
        }
    ]
    with caplog.at_level("WARNING", logger="graphrag.adapters.semantic_scholar"):
        records = parse_batch(["2401.00001"], payload)
    assert records == [ReferenceRecord("2401.00001", ("2401.00005",))]
    assert [r.getMessage() for r in caplog.records] == [
        "arXiv alias: asked 2401.00001, Semantic Scholar returned 2310.00099"
    ]


def test_matching_id_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    payload = [{"externalIds": {"ArXiv": "2401.00001v2"}, "references": []}]
    with caplog.at_level("WARNING", logger="graphrag.adapters.semantic_scholar"):
        parse_batch(["2401.00001"], payload)
    assert caplog.records == []


class Env:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []
        self.now = 0.0
        self.sleeps: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def source(self) -> SemanticScholarSource:
        return SemanticScholarSource(
            httpx.Client(transport=httpx.MockTransport(self.handler)),
            clock=lambda: self.now,
            sleep=self._sleep,
        )

    def _sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_fetch_posts_one_batch_with_arxiv_prefixed_ids() -> None:
    env = Env([httpx.Response(200, json=sample_payload())])
    records = env.source().fetch_references(IDS)
    assert len(records) == 2
    (request,) = env.requests
    assert request.method == "POST"
    assert json.loads(request.content) == {
        "ids": ["ARXIV:2401.00001", "ARXIV:9999.99999", "ARXIV:2401.00003"]
    }
    assert request.url.params["fields"] == "externalIds,references.externalIds"
    assert "x-api-key" not in request.headers  # anonymous access


def test_shared_pool_429s_are_retried_patiently() -> None:
    env = Env([httpx.Response(429)] * 3 + [httpx.Response(200, json=sample_payload())])
    assert len(env.source().fetch_references(IDS)) == 2
    assert len(env.requests) == 4
    assert env.sleeps[:3] == [10.0, 20.0, 20.0]  # capped at 20 s


def test_a_batch_gets_thirty_attempts_before_being_given_up() -> None:
    env = Env([httpx.Response(429)] * 29 + [httpx.Response(200, json=sample_payload())])
    assert len(env.source().fetch_references(IDS)) == 2
    assert len(env.requests) == 30


def test_thirty_429s_give_up_the_batch() -> None:
    env = Env([httpx.Response(429)] * 30)
    with pytest.raises(RequestFailedError, match="30 attempts"):
        env.source().fetch_references(IDS)


def test_empty_input_makes_no_request() -> None:
    env = Env([])
    assert env.source().fetch_references([]) == []
    assert env.requests == []


def test_oversized_batch_rejected() -> None:
    with pytest.raises(ValueError, match="at most"):
        Env([]).source().fetch_references([f"2401.{i:05d}" for i in range(BATCH_SIZE + 1)])
