"""Nom du conteneur Docker de la mutation : un par dossier de travail (worktrees en parallèle)."""

import re

_REFUSED = re.compile(r"[^A-Za-z0-9_.-]")


def container_name(folder: str) -> str:
    return "devgates-mutation-" + _REFUSED.sub("_", folder)
