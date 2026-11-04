from typing import Protocol

from graphrag.domain.models import Article


class ArticleReader(Protocol):
    def all_articles(self) -> list[Article]:
        """Every article of the corpus, ordered by arXiv id."""
        ...
