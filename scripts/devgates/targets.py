"""Cible de la mutation : les fichiers `src/` créés ou modifiés depuis la branche de base.

Les motifs sont ceux de `mutmut run` (`app.journal.values.*`). Si git échoue, `mutation_targets`
renvoie `None` : le gate retombe sur la mutation complète, jamais sur un saut silencieux.
"""

from devgates.mutation import Executor


def select_mutation_targets(changed: str, untracked: str) -> tuple[str, ...]:
    """Transforme les listes de fichiers de git en motifs de mutants, sans doublon."""
    patterns: list[str] = []
    for path in [*changed.splitlines(), *untracked.splitlines()]:
        if path.startswith("src/") and path.endswith(".py"):
            pattern = path.removeprefix("src/").removesuffix(".py").replace("/", ".") + ".*"
            if pattern not in patterns:
                patterns.append(pattern)
    return tuple(patterns)


def mutation_targets(git: Executor, base: str) -> tuple[str, ...] | None:
    changed = git(("git", "diff", "--name-only", base))
    untracked = git(("git", "ls-files", "--others", "--exclude-standard"))
    if changed.returncode != 0 or untracked.returncode != 0:
        return None
    return select_mutation_targets(changed.stdout, untracked.stdout)
