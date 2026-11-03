import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from datetime import date, datetime

import httpx

from graphrag.adapters.http import MinIntervalLimiter, RequestFailedError, get_with_retry
from graphrag.domain.identifiers import normalize_arxiv_id
from graphrag.domain.models import Article

API_URL = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}
CATEGORIES = ("cs.CL", "cs.IR", "cs.AI", "cs.LG")
KEYWORDS = ('"retrieval-augmented"', '"retrieval augmented"')
PAGE_SIZE = 100


def build_query(start: date, end: date) -> str:
    cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    words = " OR ".join(f"abs:{k}" for k in KEYWORDS)
    window = f"{start:%Y%m%d}0000 TO {end:%Y%m%d}2359"
    return f"({cats}) AND ({words}) AND submittedDate:[{window}]"


def _is_error_body(response: httpx.Response) -> bool:
    return "arxiv.org/api/errors" in response.text[:2000]


def parse_page(xml_text: str) -> tuple[int, list[Article]]:
    """Return (total results announced by the server, articles of this page)."""
    root = ET.fromstring(xml_text)
    total = int(root.findtext("os:totalResults", namespaces=_NS) or 0)
    articles = []
    for entry in root.findall("a:entry", _NS):
        entry_id = entry.findtext("a:id", namespaces=_NS) or ""
        if entry_id.endswith("/api/errors"):
            raise RequestFailedError("arXiv returned an error entry")
        articles.append(
            Article(
                arxiv_id=normalize_arxiv_id(entry_id),
                title=_clean(entry.findtext("a:title", namespaces=_NS)),
                abstract=_clean(entry.findtext("a:summary", namespaces=_NS)),
                published=_parse_date(entry.findtext("a:published", namespaces=_NS)),
                categories=tuple(c.get("term", "") for c in entry.findall("a:category", _NS)),
                author_names=tuple(
                    _clean(a.findtext("a:name", namespaces=_NS))
                    for a in entry.findall("a:author", _NS)
                ),
            )
        )
    return total, articles


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _parse_date(text: str | None) -> date:
    if not text:
        raise ValueError("arXiv entry without a published date")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).date()


class ArxivSource:
    def __init__(
        self,
        client: httpx.Client,
        *,
        min_interval: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self._client = client
        self._limiter = MinIntervalLimiter(min_interval, clock, sleep)
        self._sleep = sleep
        self._page_size = page_size

    def iter_papers(self, start: date, end: date) -> Iterator[Article]:
        query = build_query(start, end)
        offset = 0
        total = None
        while total is None or offset < total:
            response = get_with_retry(
                self._client,
                API_URL,
                {
                    "search_query": query,
                    "start": offset,
                    "max_results": self._page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": "ascending",
                },
                limiter=self._limiter,
                sleep=self._sleep,
                is_error_body=_is_error_body,
            )
            total, articles = parse_page(response.text)
            if not articles:
                break
            yield from articles
            offset += len(articles)
