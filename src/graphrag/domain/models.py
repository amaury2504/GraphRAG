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
class IndexedChunk:
    """A chunk ready to store: `search_text` (title + chunk) is what gets embedded and indexed."""

    chunk: Chunk
    search_text: str
    embedding: tuple[float, ...]

    @property
    def key(self) -> str:
        return f"{self.chunk.arxiv_id}#{self.chunk.position}"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One chunk found by a search, with its article; `score` depends on the search used."""

    arxiv_id: str
    title: str
    position: int
    text: str
    score: float
    abstract: str = ""  # the whole abstract of the article; `text` is only the matching chunk


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens consumed by language model calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    usage: Usage


@dataclass(frozen=True, slots=True)
class Timings:
    """Seconds spent in each stage of answering a question."""

    retrieval_seconds: float = 0.0
    llm_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.llm_seconds


@dataclass(frozen=True, slots=True)
class Answer:
    """A written answer. `citations` are the sources the text cites, in order of first mention."""

    question: str
    text: str
    citations: tuple[SearchHit, ...]
    retrieved: tuple[SearchHit, ...]
    usage: Usage
    timings: Timings = Timings()


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
