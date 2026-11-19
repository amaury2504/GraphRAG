from collections.abc import Iterator, Sequence
from datetime import date
from typing import Protocol

from graphrag.domain.models import (
    Article,
    ArticleChanges,
    GraphCounts,
    ReferenceRecord,
    WorkRecord,
)


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

    def upsert_articles(self, articles: Sequence[Article]) -> ArticleChanges:
        """Insert new articles and rewrite those whose hash differs from the stored one.

        Articles with the same hash are left as they are. Returns how many were new,
        modified and unchanged. Authors, citations and chunks are not touched here.
        """
        ...

    def upsert_works(self, records: Sequence[WorkRecord]) -> None:
        """Attach OpenAlex ids and make AUTHORED_BY match each record exactly. Idempotent.

        Relations to authors a record no longer lists are removed, then authors left with
        no article are deleted, in the same transaction.
        """
        ...

    def upsert_citations(self, records: Sequence[ReferenceRecord]) -> None:
        """Make the outgoing CITES of each record match it, corpus articles only. Idempotent.

        Citations a record no longer lists are removed; an empty list removes them all.
        """
        ...

    def delete_article(self, arxiv_id: str) -> bool:
        """Remove an article with its chunks, its CITES (both ways) and its orphan authors.

        Returns False if the article was not stored.
        """
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

    def forget(self, arxiv_ids: Sequence[str]) -> None:
        """Make ids to be looked up again, e.g. because their article was modified."""
        ...
