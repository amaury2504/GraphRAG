import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "graphrag"
FORBIDDEN_IN_DOMAIN = {
    "langchain",
    "langchain_core",
    "langgraph",
    "neo4j",
    "fastapi",
    "psycopg",
    "pydantic_settings",
}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_domain_imports_no_framework() -> None:
    offenders = {
        str(p.relative_to(SRC)): bad
        for p in (SRC / "domain").rglob("*.py")
        if (bad := _imports(p) & FORBIDDEN_IN_DOMAIN)
    }
    assert offenders == {}


@pytest.mark.parametrize("package", ["domain", "ports", "pipeline", "api", "cli"])
def test_langchain_only_in_adapters(package: str) -> None:
    offenders = {
        str(p.relative_to(SRC)): bad
        for p in (SRC / package).rglob("*.py")
        if (bad := {m for m in _imports(p) if m.startswith("langchain")})
    }
    assert offenders == {}


def test_langgraph_unused_in_v1() -> None:
    offenders = [str(p) for p in SRC.rglob("*.py") if "langgraph" in _imports(p)]
    assert offenders == []


def test_architecture_scan_is_not_vacuous() -> None:
    # Guards against the scan silently finding no files (e.g. wrong path).
    assert len(list(SRC.rglob("*.py"))) > 5
    assert "neo4j" in _imports(SRC / "adapters" / "health.py")
