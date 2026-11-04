import pytest

from graphrag.adapters.neo4j_store import lucene_query


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Graph RAG", "graph rag"),
        ("multi-hop QA?", "multi\\-hop qa\\?"),
        ('"exact" (phrase)', '\\"exact\\" \\(phrase\\)'),
        ("a && b || !c", "a \\&\\& b \\|\\| \\!c"),
        (
            "title:x^2 ~fuzzy* [1 TO 5] {y} a/b c\\d",
            ("title\\:x\\^2 \\~fuzzy\\* \\[1 to 5\\] \\{y\\} a\\/b c\\\\d"),
        ),
        ("RAG AND NOT graphs OR trees", "rag and not graphs or trees"),
        ("  spaced \n out  ", "spaced out"),
    ],
)
def test_lucene_query_escapes_operators(text: str, expected: str) -> None:
    assert lucene_query(text) == expected
