"""Suivi de la CI d'une PR : avancement par job et par étape, pourcentage de complétion.

Usage : uv run python scripts/ci_watch.py [--branch B] [--sha S] [--interval 10] [--timeout 1800]

Cherche le run de CI du commit exact (HEAD par défaut), puis l'interroge jusqu'à sa fin. N'affiche
une ligne que lorsque l'état change (pratique pour `Monitor`). Code de sortie : 0 CI verte,
1 CI rouge ou annulée, 2 run introuvable ou délai dépassé.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from typing import TypedDict

GH_WINDOWS = r"C:\Program Files\GitHub CLI\gh.exe"
FAIT = {"completed"}
ICONES = {"success": "OK", "failure": "KO", "cancelled": "--", "skipped": "--"}


def trouver_gh() -> str:
    return shutil.which("gh") or GH_WINDOWS


class Etape(TypedDict):
    name: str
    status: str
    conclusion: str | None


class Job(TypedDict):
    name: str
    status: str
    conclusion: str | None
    steps: list[Etape]


class Run(TypedDict):
    status: str
    conclusion: str | None
    jobs: list[Job]


class RunResume(TypedDict):
    databaseId: int
    headSha: str


def gh(args: list[str]) -> str:
    return subprocess.run(
        [trouver_gh(), *args], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.strip()


def barre(pourcentage: int, largeur: int = 20) -> str:
    plein = round(largeur * pourcentage / 100)
    return "#" * plein + "." * (largeur - plein)


def pourcentage(etapes: list[Etape]) -> int:
    if not etapes:
        return 0
    finies = sum(1 for e in etapes if e["status"] in FAIT)
    return round(100 * finies / len(etapes))


def rendre(run: Run) -> list[str]:
    """Lignes d'affichage d'un run : une par job, une par étape en cours ou échouée."""
    lignes: list[str] = []
    toutes = [e for j in run["jobs"] for e in j["steps"]]
    lignes.append(f"CI {run['status']} [{barre(pourcentage(toutes))}] {pourcentage(toutes)} %")
    for job in run["jobs"]:
        etapes = job["steps"]
        etat = ICONES.get(job["conclusion"] or "", job["status"])
        finies = sum(1 for e in etapes if e["status"] in FAIT)
        lignes.append(
            f"  {job['name']}: {etat} [{barre(pourcentage(etapes))}] "
            f"{pourcentage(etapes)} % ({finies}/{len(etapes)} étapes)"
        )
        for e in etapes:
            if e["status"] == "in_progress" or e["conclusion"] == "failure":
                marque = "en cours" if e["status"] == "in_progress" else "ECHEC"
                lignes.append(f"    - {e['name']}: {marque}")
    return lignes


def trouver_run(branche: str, sha: str, fin: float, intervalle: int) -> int | None:
    while time.monotonic() < fin:
        runs: list[RunResume] = json.loads(
            gh(
                [
                    "run",
                    "list",
                    "--branch",
                    branche,
                    "--limit",
                    "10",
                    "--json",
                    "databaseId,headSha",
                ]
            )
        )
        for r in runs:
            if r["headSha"] == sha:
                return int(r["databaseId"])
        time.sleep(intervalle)
    return None


def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", default=None)
    p.add_argument("--sha", default=None)
    p.add_argument("--interval", type=int, default=10)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--live", action="store_true", help="redessine l'écran à chaque changement")
    a = p.parse_args()
    if a.live:
        os.system("")  # active les séquences ANSI de la console Windows

    branche = a.branch or git("rev-parse", "--abbrev-ref", "HEAD")
    sha = a.sha or git("rev-parse", "HEAD")
    fin = time.monotonic() + a.timeout

    print(f"Suivi CI : {branche} @ {sha[:7]}", flush=True)
    run_id = trouver_run(branche, sha, fin, a.interval)
    if run_id is None:
        print("Aucun run de CI trouvé pour ce commit (délai dépassé).", flush=True)
        return 2

    dernier: list[str] = []
    while time.monotonic() < fin:
        run: Run = json.loads(gh(["run", "view", str(run_id), "--json", "status,conclusion,jobs"]))
        lignes = rendre(run)
        if lignes != dernier:
            if a.live:  # redessine l'écran : un seul tableau qui avance
                print("\x1b[2J\x1b[H" + f"Suivi CI : {branche} @ {sha[:7]}\n", flush=True)
            print("\n".join(lignes), flush=True)
            dernier = lignes
        if run["status"] == "completed":
            verte = run["conclusion"] == "success"
            print("CI VERTE" if verte else f"CI ROUGE ({run['conclusion']})", flush=True)
            return 0 if verte else 1
        time.sleep(a.interval)
    print("Délai dépassé avant la fin de la CI.", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
