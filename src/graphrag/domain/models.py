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
class Chunk:
    """A piece of an article's abstract, the unit that gets embedded and searched."""

    arxiv_id: str
    position: int
    text: str


@dataclass(frozen=True, slots=True)
class Author:
    key: str
    name: str


@dataclass(frozen=True, slots=True)
class WorkRecord:
    """What the author source knows about one article of the corpus."""

    arxiv_id: str
    openalex_id: str
    authors: tuple[Author, ...]


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    """The arXiv articles cited by one article (only references that have an arXiv id)."""

    arxiv_id: str
    cited_arxiv_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphCounts:
    articles: int
    authors: int
    authored_by: int
    cites: int
