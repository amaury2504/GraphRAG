from collections.abc import Iterator, Sequence
from datetime import date
from typing import Protocol

from graphrag.domain.models import Article, GraphCounts, WorkRecord


class PaperSource(Protocol):
    def iter_papers(self, start: date, end: date) -> Iterator[Article]:
        """Yield every matching article submitted between `start` and `end` (inclusive)."""
        ...


class CitationSource(Protocol):
    def fetch_works(self, arxiv_ids: Sequence[str]) -> list[WorkRecord]:
        """Return records for the ids the source knows; unknown ids are simply absent."""
        ...


class GraphStore(Protocol):
    def ensure_schema(self) -> None: ...

    def upsert_articles(self, articles: Sequence[Article]) -> None:
        """Insert articles not yet present; existing ones are left untouched."""
        ...

    def upsert_works(self, records: Sequence[WorkRecord]) -> None:
        """Attach OpenAlex ids, authors and AUTHORED_BY relations. Idempotent."""
        ...

    def upsert_citations(self, records: Sequence[WorkRecord]) -> None:
        """Create CITES relations between corpus articles only. Idempotent."""
        ...

    def article_ids(self) -> list[str]: ...

    def counts(self) -> GraphCounts: ...
