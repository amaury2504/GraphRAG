from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Article:
    arxiv_id: str
    title: str
    abstract: str
    published: date
    categories: tuple[str, ...]
    author_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Author:
    key: str
    name: str


@dataclass(frozen=True, slots=True)
class WorkRecord:
    """What the citation source knows about one article of the corpus."""

    arxiv_id: str
    openalex_id: str
    authors: tuple[Author, ...]
    referenced_openalex_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphCounts:
    articles: int
    authors: int
    authored_by: int
    cites: int
