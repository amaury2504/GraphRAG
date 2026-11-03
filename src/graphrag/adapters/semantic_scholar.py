import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from graphrag.adapters.http import MinIntervalLimiter, send_with_retry
from graphrag.domain.identifiers import normalize_arxiv_id
from graphrag.domain.models import ReferenceRecord

API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "externalIds,references.externalIds"
BATCH_SIZE = 100


class SemanticScholarSource:
    """Reference lists from the Semantic Scholar Graph API, without an API key.

    The anonymous pool is shared and often answers 429, with short open windows: retries
    are frequent (at most 20 s apart) and many (30), so a batch catches a window without
    being given up too early (about 10 minutes of waiting in total).
    """

    def __init__(
        self,
        client: httpx.Client,
        *,
        min_interval: float = 1.0,
        retry_delay: float = 10.0,
        max_delay: float = 20.0,
        max_attempts: int = 30,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._limiter = MinIntervalLimiter(min_interval, clock, sleep)
        self._retry_delay = retry_delay
        self._max_delay = max_delay
        self._max_attempts = max_attempts
        self._sleep = sleep

    def fetch_references(self, arxiv_ids: Sequence[str]) -> list[ReferenceRecord]:
        if len(arxiv_ids) > BATCH_SIZE:
            raise ValueError(f"at most {BATCH_SIZE} ids per call, got {len(arxiv_ids)}")
        if not arxiv_ids:
            return []
        response = send_with_retry(
            self._client,
            "POST",
            API_URL,
            {"fields": FIELDS},
            json={"ids": [f"ARXIV:{i}" for i in arxiv_ids]},
            limiter=self._limiter,
            max_attempts=self._max_attempts,
            base_delay=self._retry_delay,
            max_delay=self._max_delay,
            sleep=self._sleep,
        )
        return parse_batch(arxiv_ids, response.json())


def parse_batch(arxiv_ids: Sequence[str], payload: list[Any]) -> list[ReferenceRecord]:
    """Results come back in request order, with null for unknown papers."""
    if len(payload) != len(arxiv_ids):
        raise ValueError(f"asked for {len(arxiv_ids)} papers, got {len(payload)} results")
    records = []
    for requested, item in zip(arxiv_ids, payload, strict=True):
        if item is None:
            continue
        returned = (item.get("externalIds") or {}).get("ArXiv")
        if returned is not None and _safe_normalize(returned) != requested:
            raise ValueError(f"result order mismatch: asked {requested}, got {returned}")
        cited: dict[str, None] = {}  # ordered, de-duplicated
        for reference in item.get("references") or []:
            raw = ((reference or {}).get("externalIds") or {}).get("ArXiv")
            if raw and (cited_id := _safe_normalize(raw)) and cited_id != requested:
                cited[cited_id] = None
        records.append(ReferenceRecord(requested, tuple(cited)))
    return records


def _safe_normalize(raw: str) -> str | None:
    try:
        return normalize_arxiv_id(raw)
    except ValueError:
        return None
