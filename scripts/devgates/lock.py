"""Verrou par dossier de travail : deux lancements des gates au même endroit se détruiraient
mutuellement (conteneur, dossier `mutants`). Le verrou contient le PID du lancement en cours."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


class RunLockedError(Exception):
    """Un autre lancement des gates est en cours dans ce dossier."""


@contextmanager
def run_lock(path: Path, pid: int, is_alive: Callable[[int], bool]) -> Iterator[None]:
    path.parent.mkdir(exist_ok=True)
    holder = _holder(path)
    if holder is not None and is_alive(holder):
        raise RunLockedError(f"gates déjà en cours dans ce dossier (processus {holder})")
    path.write_text(str(pid), encoding="utf-8")
    try:
        yield
    finally:
        path.unlink()


def _holder(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_bytes())  # octets : pas de choix d'encodage à muter
    except ValueError:
        return None  # verrou illisible : laissé par un lancement interrompu
