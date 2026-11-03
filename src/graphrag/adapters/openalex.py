import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from graphrag.adapters.http import get_with_retry
from graphrag.domain.identifiers import arxiv_doi_url, arxiv_id_from_doi, short_openalex_id
from graphrag.domain.models import Author, WorkRecord

API_URL = "https://api.openalex.org/works"
BATCH_SIZE = 100  # OpenAlex accepts at most 100 OR values per filter


class OpenAlexSource:
    def __init__(
        self,
        client: httpx.Client,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._sleep = sleep

    def fetch_works(self, arxiv_ids: Sequence[str]) -> list[WorkRecord]:
        if len(arxiv_ids) > BATCH_SIZE:
            raise ValueError(f"at most {BATCH_SIZE} ids per call, got {len(arxiv_ids)}")
        if not arxiv_ids:
            return []
        response = get_with_retry(
            self._client,
            API_URL,
            {
                "filter": "doi:" + "|".join(arxiv_doi_url(i) for i in arxiv_ids),
                "select": "id,doi,authorships,referenced_works",
                "per_page": BATCH_SIZE,
            },
            sleep=self._sleep,
        )
        return parse_works(response.json())


def parse_works(payload: dict[str, Any]) -> list[WorkRecord]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("unexpected OpenAlex payload: no results list")
    records: dict[str, WorkRecord] = {}
    for work in results:
        arxiv_id = arxiv_id_from_doi(work.get("doi") or "")
        if arxiv_id is None:
            continue
        authors = tuple(
            Author(short_openalex_id(a["id"]), a.get("display_name") or "")
            for authorship in work.get("authorships") or []
            if (a := authorship.get("author") or {}).get("id")
        )
        records.setdefault(
            arxiv_id,
            WorkRecord(
                arxiv_id=arxiv_id,
                openalex_id=short_openalex_id(work["id"]),
                authors=authors,
                referenced_openalex_ids=tuple(
                    short_openalex_id(r) for r in work.get("referenced_works") or []
                ),
            ),
        )
    return list(records.values())
