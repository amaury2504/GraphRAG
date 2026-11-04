"""Neo4j store and ingestion against the real database.

Only touches nodes whose ids start with the TEST prefixes below, and removes them afterwards,
so it is safe to run next to a real corpus.
"""

import os
from collections.abc import Iterator, Sequence
from datetime import date

import neo4j
import pytest

from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.cli.ingest import run_ingestion
from graphrag.domain.models import Article, Author, GraphCounts, ReferenceRecord, WorkRecord

pytestmark = pytest.mark.integration

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")


def _password() -> str:
    value = os.environ.get("NEO4J_PASSWORD")
    if not value:
        pytest.fail("NEO4J_PASSWORD must be set to run integration tests")
    return value


def _cleanup() -> None:
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        driver.execute_query("MATCH (c:Chunk) WHERE c.arxiv_id STARTS WITH '9999.' DETACH DELETE c")
        driver.execute_query(
            "MATCH (a:Article) WHERE a.arxiv_id STARTS WITH '9999.' DETACH DELETE a"
        )
        driver.execute_query("MATCH (p:Author) WHERE p.key STARTS WITH 'ATEST' DETACH DELETE p")


@pytest.fixture
def store() -> Iterator[Neo4jStore]:
    _cleanup()
    with Neo4jStore(URI, USER, _password()) as s:
        s.ensure_schema()
        yield s
    _cleanup()


def art(n: int, title: str = "T") -> Article:
    return Article(
        f"9999.{n:05d}",
        f"{title}{n}",
        "abstract",
        date(2025, 3, 4),
        ("cs.CL", "cs.IR"),
        ("Ann", "Bob"),
    )


A, B, C = art(1), art(2), art(3)
WORKS = [
    WorkRecord(A.arxiv_id, "WTEST1", (Author("ATEST1", "Ann"),)),
    WorkRecord(B.arxiv_id, "WTEST2", (Author("ATEST1", "Ann"), Author("ATEST2", "Bob"))),
]
# A cites B, an out-of-corpus paper, and itself (must be ignored)
REFERENCES = [ReferenceRecord(A.arxiv_id, (B.arxiv_id, "9998.00001", A.arxiv_id))]


def delta(before: GraphCounts, after: GraphCounts) -> tuple[int, int, int, int]:
    return (
        after.articles - before.articles,
        after.authors - before.authors,
        after.authored_by - before.authored_by,
        after.cites - before.cites,
    )


def test_schema_creation_is_repeatable(store: Neo4jStore) -> None:
    store.ensure_schema()
    store.ensure_schema()


def test_articles_are_stored_with_all_properties(store: Neo4jStore) -> None:
    store.upsert_articles([A])
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        records, _, _ = driver.execute_query(
            "MATCH (a:Article {arxiv_id: $id}) RETURN a", id=A.arxiv_id
        )
    node = records[0]["a"]
    assert node["title"] == "T1"
    assert node["abstract"] == "abstract"
    assert node["published"].to_native() == date(2025, 3, 4)
    assert node["categories"] == ["cs.CL", "cs.IR"]
    assert node["author_names"] == ["Ann", "Bob"]


def test_existing_article_is_not_reinserted_or_modified(store: Neo4jStore) -> None:
    before = store.counts()
    store.upsert_articles([art(1, "original")])
    store.upsert_articles([art(1, "changed")])
    assert delta(before, store.counts())[0] == 1
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        records, _, _ = driver.execute_query(
            "MATCH (a:Article {arxiv_id: '9999.00001'}) RETURN a.title AS t"
        )
    assert records[0]["t"] == "original1"


def test_works_and_citations_build_expected_graph(store: Neo4jStore) -> None:
    before = store.counts()
    store.upsert_articles([A, B, C])
    store.upsert_works(WORKS)
    store.upsert_citations(REFERENCES)
    # 3 articles; authors ATEST1 shared by 2 articles; 3 AUTHORED_BY; 1 CITES (A->B only)
    assert delta(before, store.counts()) == (3, 2, 3, 1)
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        records, _, _ = driver.execute_query(
            "MATCH (s:Article)-[:CITES]->(t:Article) WHERE s.arxiv_id STARTS WITH '9999.' "
            "RETURN s.arxiv_id AS s, t.arxiv_id AS t"
        )
        order, _, _ = driver.execute_query(
            "MATCH (:Article {arxiv_id: '9999.00002'})-[r:AUTHORED_BY]->(p) "
            "RETURN p.key AS k ORDER BY r.position"
        )
    assert [(r["s"], r["t"]) for r in records] == [(A.arxiv_id, B.arxiv_id)]
    assert [r["k"] for r in order] == ["ATEST1", "ATEST2"]


def test_writing_twice_adds_nothing(store: Neo4jStore) -> None:
    def write() -> None:
        store.upsert_articles([A, B, C])
        store.upsert_works(WORKS)
        store.upsert_citations(REFERENCES)

    before = store.counts()
    write()
    first = store.counts()
    write()
    assert store.counts() == first
    assert delta(before, first) == (3, 2, 3, 1)


def test_batches_larger_than_write_size(store: Neo4jStore) -> None:
    many = [art(n) for n in range(1, 1201)]
    before = store.counts()
    store.upsert_articles(many)
    assert delta(before, store.counts())[0] == 1200


class FakePapers:
    def __init__(self, fail_after: int | None = None) -> None:
        self.fail_after = fail_after

    def iter_papers(self, start: date, end: date) -> Iterator[Article]:
        for i, a in enumerate((A, B, C)):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("arXiv went away")
            yield a


class FakeAuthors:
    def fetch_works(self, arxiv_ids: Sequence[str]) -> list[WorkRecord]:
        return [r for r in WORKS if r.arxiv_id in arxiv_ids]


class FakeCitations:
    def fetch_references(self, arxiv_ids: Sequence[str]) -> list[ReferenceRecord]:
        return [r for r in REFERENCES if r.arxiv_id in arxiv_ids]


def ingest(store: Neo4jStore, papers: FakePapers) -> GraphCounts:
    report = run_ingestion(
        papers,
        FakeAuthors(),
        FakeCitations(),
        store,
        date(2025, 1, 1),
        date(2025, 12, 31),
        log=lambda _m: None,
    )
    return report.counts


def test_acceptance_second_run_adds_no_duplicates(store: Neo4jStore) -> None:
    before = store.counts()
    first = ingest(store, FakePapers())
    second = ingest(store, FakePapers())
    assert first == second
    assert delta(before, second) == (3, 2, 3, 1)


def test_interrupted_run_then_rerun_matches_clean_run(store: Neo4jStore) -> None:
    before = store.counts()
    with pytest.raises(RuntimeError, match="went away"):
        ingest(store, FakePapers(fail_after=2))
    final = ingest(store, FakePapers())
    assert delta(before, final) == (3, 2, 3, 1)
