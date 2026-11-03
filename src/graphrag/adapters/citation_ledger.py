from collections.abc import Sequence
from pathlib import Path

DEFAULT_PATH = Path("data/citations_checked.txt")


class FileCitationLedger:
    """One arXiv id per line in a local file kept out of the graph (and out of git)."""

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._path = path

    def checked(self) -> set[str]:
        if not self._path.exists():
            return set()
        lines = self._path.read_text(encoding="utf-8").splitlines()
        return {line.strip() for line in lines if line.strip()}

    def mark(self, arxiv_ids: Sequence[str]) -> None:
        if not arxiv_ids:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.writelines(f"{i}\n" for i in arxiv_ids)
            handle.flush()
