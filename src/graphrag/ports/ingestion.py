from collections.abc import Iterator, Sequence
from datetime import date
from typing import Protocol

from graphrag.domain.models import Article, GraphCounts, ReferenceRecord, WorkRecord


class PaperSource(Protocol):
    def iter_papers(self, start: date, end: date) -> Iterator[Article]:
        """Yield every matching article submitted between `start` and `end` (inclusive)."""
        ...


class AuthorSource(Protocol):
    def fetch_works(self, arxiv_ids: Sequence[str]) -> list[WorkRecord]:
        """Return records for the ids the source knows; unknown ids are simply absent."""
        ...


class CitationSource(Protocol):
    def fetch_references(self, arxiv_ids: Sequence[str]) -> list[ReferenceRecord]:
        """Return the cited arXiv ids of each known article; unknown ids are simply absent."""
        ...


class GraphStore(Protocol):
    def ensure_schema(self) -> None: ...

    def upsert_articles(self, articles: Sequence[Article]) -> None:
        """Insert articles not yet present; existing ones are left untouched."""
        ...

    def upsert_works(self, records: Sequence[WorkRecord]) -> None:
        """Attach OpenAlex ids, authors and AUTHORED_BY relations. Idempotent."""
        ...

    def upsert_citations(self, records: Sequence[ReferenceRecord]) -> None:
        """Create CITES relations between corpus articles only. Idempotent."""
        ...

    def article_ids(self) -> list[str]: ...

    def counts(self) -> GraphCounts: ...


class CitationLedger(Protocol):
    """Remembers which articles already had their references looked up, across runs."""

    def checked(self) -> set[str]:
        """Ids whose lookup completed and whose citations were written."""
        ...

    def mark(self, arxiv_ids: Sequence[str]) -> None:
        """Record ids as done; call only after their citations are stored."""
        ...
