import re

from graphrag.domain.models import Chunk

DEFAULT_MAX_WORDS = 200

# A sentence ends with . ! or ? followed by whitespace and something that looks like a new
# sentence start. The lookahead avoids cutting on "e.g. the" or "3.5 points".
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences after collapsing all whitespace (arXiv abstracts have newlines)."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return _SENTENCE_END.split(normalized)


TEXT_TEMPLATE = "title. chunk"  # what `search_text` builds; recorded with the index


def search_text(title: str, chunk_text: str) -> str:
    """The text that gets embedded and keyword-indexed: the article title, then the chunk.

    The title gives context to chunks that no longer name the topic ("We evaluate on...").
    """
    clean_title = " ".join(title.split())
    if not clean_title:
        return chunk_text
    if clean_title[-1] not in ".!?":
        clean_title += "."
    return f"{clean_title} {chunk_text}"


def chunk_abstract(
    arxiv_id: str, abstract: str, max_words: int = DEFAULT_MAX_WORDS
) -> tuple[Chunk, ...]:
    """Cut an abstract into chunks of at most `max_words` words, on sentence boundaries.

    An abstract that fits is a single chunk. A longer one is split greedily, and each chunk
    after the first repeats the last sentence of the previous one (overlap) when it fits.
    A single sentence longer than `max_words` is cut by words.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")

    pieces: list[str] = []
    for sentence in split_sentences(abstract):
        words = sentence.split()
        if len(words) <= max_words:
            pieces.append(sentence)
        else:
            pieces.extend(
                " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)
            )

    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for piece in pieces:
        n = len(piece.split())
        if current and current_words + n > max_words:
            groups.append(current)
            overlap = current[-1]
            if len(overlap.split()) + n <= max_words:
                current, current_words = [overlap], len(overlap.split())
            else:
                current, current_words = [], 0
        current.append(piece)
        current_words += n
    if current:
        groups.append(current)

    return tuple(
        Chunk(arxiv_id=arxiv_id, position=i, text=" ".join(group)) for i, group in enumerate(groups)
    )
