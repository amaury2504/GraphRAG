from datetime import date
from pathlib import Path

import httpx
import pytest

from graphrag.adapters.arxiv import ArxivSource, build_query, parse_page
from graphrag.adapters.http import RequestFailedError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "arxiv_page.xml"

ERROR_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry><id>https://arxiv.org/api/errors</id><title>Error</title></entry>
</feed>"""


def page_xml(total: int, ids: list[str]) -> str:
    entries = "".join(
        f"""<entry><id>http://arxiv.org/abs/{i}v2</id><title> T {i} </title>
        <summary>S</summary><published>2025-10-01T02:51:58Z</published>
        <category term="cs.CL"/><author><name>A</name></author></entry>"""
        for i in ids
    )
    return (
        "<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom' "
        "xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>"
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>"
    )


def test_parse_real_page() -> None:
    total, articles = parse_page(FIXTURE.read_text(encoding="utf-8"))
    assert total > 3
    assert len(articles) == 3
    first = articles[0]
    assert first.arxiv_id == "2510.00444"  # version suffix stripped
    assert first.title.startswith("TokMem: One-Token Procedural Memory")
    assert first.published == date(2025, 10, 1)
    assert first.categories[0] == "cs.CL"
    assert first.author_names[:3] == ("Zijun Wu", "Yongchang Hao", "Lili Mou")
    for article in articles:
        assert "\n" not in article.title
        assert "\n" not in article.abstract
        assert article.abstract
        assert article.author_names
        assert all(c for c in article.categories)


def test_error_entry_raises() -> None:
    with pytest.raises(RequestFailedError):
        parse_page(ERROR_XML)


def test_build_query_contains_filters_and_window() -> None:
    q = build_query(date(2025, 11, 1), date(2026, 10, 31))
    assert "cat:cs.CL" in q
    assert "cat:cs.LG" in q
    assert 'abs:"retrieval-augmented"' in q
    assert "submittedDate:[202511010000 TO 202610312359]" in q


class Recorder:
    def __init__(self, bodies: list[str]) -> None:
        self.bodies = bodies
        self.requests: list[httpx.Request] = []
        self.now = 0.0
        self.sleeps: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, text=self.bodies.pop(0))

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s

    def source(self, page_size: int = 2) -> ArxivSource:
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        return ArxivSource(client, clock=self.clock, sleep=self.sleep, page_size=page_size)


def test_pagination_walks_all_pages_and_respects_rate_limit() -> None:
    rec = Recorder(
        [page_xml(5, ["2501.00001", "2501.00002"]), page_xml(5, ["2501.00003", "2501.00004"]),
         page_xml(5, ["2501.00005"])]
    )  # fmt: skip
    ids = [a.arxiv_id for a in rec.source().iter_papers(date(2025, 1, 1), date(2025, 12, 31))]
    assert ids == [f"2501.0000{i}" for i in range(1, 6)]
    starts = [r.url.params["start"] for r in rec.requests]
    assert starts == ["0", "2", "4"]
    assert all(r.url.params["max_results"] == "2" for r in rec.requests)
    assert rec.sleeps == [3.0, 3.0]  # 3 s between the 3 requests


def test_empty_result_yields_nothing() -> None:
    rec = Recorder([page_xml(0, [])])
    assert list(rec.source().iter_papers(date(2025, 1, 1), date(2025, 1, 2))) == []
    assert len(rec.requests) == 1


def test_stops_if_server_returns_empty_page_before_total() -> None:
    rec = Recorder([page_xml(10, ["2501.00001"]), page_xml(10, [])])
    ids = [a.arxiv_id for a in rec.source().iter_papers(date(2025, 1, 1), date(2025, 1, 2))]
    assert ids == ["2501.00001"]  # no infinite loop


def test_transient_error_body_is_retried_then_succeeds() -> None:
    rec = Recorder([ERROR_XML, page_xml(1, ["2501.00001"])])
    ids = [a.arxiv_id for a in rec.source().iter_papers(date(2025, 1, 1), date(2025, 1, 2))]
    assert ids == ["2501.00001"]
    assert len(rec.requests) == 2
