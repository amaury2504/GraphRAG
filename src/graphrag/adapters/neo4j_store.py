from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from typing import Any

import neo4j

from graphrag.domain.hashing import article_hash, content_hash
from graphrag.domain.models import (
    Article,
    ArticleChanges,
    ArticleIndexState,
    GraphCounts,
    IndexedChunk,
    IndexInfo,
    ReferenceRecord,
    SearchHit,
    WorkRecord,
)
from graphrag.domain.versioning import VersionInfo

WRITE_BATCH = 500

SCHEMA = (
    "CREATE CONSTRAINT article_arxiv_id IF NOT EXISTS FOR (a:Article) REQUIRE a.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT author_key IF NOT EXISTS FOR (a:Author) REQUIRE a.key IS UNIQUE",
)

STORED_ARTICLE_HASHES = """
MATCH (a:Article) WHERE a.arxiv_id IN $ids
RETURN a.arxiv_id AS id, a.content_hash AS hash, a.title AS title, a.abstract AS abstract
"""

# Rewrites the fields the source provides; authors, citations and chunks are left alone.
UPSERT_ARTICLES = """
UNWIND $rows AS row
MERGE (a:Article {arxiv_id: row.arxiv_id})
SET a.title = row.title,
    a.abstract = row.abstract,
    a.published = date(row.published),
    a.categories = row.categories,
    a.author_names = row.author_names,
    a.content_hash = row.content_hash
"""

UPSERT_WORKS = """
UNWIND $rows AS row
MATCH (a:Article {arxiv_id: row.arxiv_id})
SET a.openalex_id = row.openalex_id
WITH a, row
UNWIND row.authors AS author
MERGE (p:Author {key: author.key})
ON CREATE SET p.name = author.name
MERGE (a)-[r:AUTHORED_BY]->(p)
SET r.position = author.position
"""

# Run after UPSERT_WORKS: relations to authors a record no longer lists, then the authors
# left with no article (only those just detached, never a global sweep).
PRUNE_AUTHORS = """
UNWIND $rows AS row
MATCH (:Article {arxiv_id: row.arxiv_id})-[r:AUTHORED_BY]->(p:Author)
WHERE NOT p.key IN [author IN row.authors | author.key]
DELETE r
WITH DISTINCT p
WHERE NOT EXISTS { (p)<-[:AUTHORED_BY]-() }
DELETE p
"""

# Run before UPSERT_CITATIONS: an empty `cited` list removes every outgoing CITES.
PRUNE_CITATIONS = """
UNWIND $rows AS row
MATCH (:Article {arxiv_id: row.arxiv_id})-[r:CITES]->(t:Article)
WHERE NOT t.arxiv_id IN row.cited
DELETE r
"""

AUTHORS_OF_ARTICLE = """
MATCH (a:Article {arxiv_id: $id})
OPTIONAL MATCH (a)-[:AUTHORED_BY]->(p:Author)
RETURN count(DISTINCT a) AS n, collect(p.key) AS keys
"""

# Chunks are immutable and shared with older versions: they are only detached from the article
# (so no search finds them) and left for the garbage collection (T4).
DELETE_ARTICLE = """
MATCH (a:Article {arxiv_id: $id})
DETACH DELETE a
"""

DELETE_ORPHAN_AUTHORS = """
MATCH (p:Author)
WHERE p.key IN $keys AND NOT EXISTS { (p)<-[:AUTHORED_BY]-() }
DELETE p
"""

UPSERT_CITATIONS = """
UNWIND $rows AS row
MATCH (s:Article {arxiv_id: row.arxiv_id})
UNWIND row.cited AS cited_id
MATCH (t:Article {arxiv_id: cited_id})
WHERE t <> s
MERGE (s)-[:CITES]->(t)
"""


VECTOR_INDEX = "chunk_embedding"
FULLTEXT_INDEX = "chunk_text"

INDEX_INFO_CONSTRAINT = (
    "CREATE CONSTRAINT index_info_name IF NOT EXISTS FOR (i:IndexInfo) REQUIRE i.name IS UNIQUE"
)
CHUNK_KEY_CONSTRAINT = (
    "CREATE CONSTRAINT chunk_key IF NOT EXISTS FOR (c:Chunk) REQUIRE c.key IS UNIQUE"
)


def create_vector_index(dimensions: int) -> str:
    """Schema commands take the size as a literal; callers pass a checked int."""
    return f"""
CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {dimensions},
  `vector.similarity_function`: 'cosine'
}}}}
"""


# "english" drops stop words and reduces words to their stem ("agents" matches "agent").
# `IF NOT EXISTS` is a no-op when an index on the same label and property exists, whatever
# its options, so a different analyzer is only applied by dropping the index first.
FULLTEXT_ANALYZER = "english"
CREATE_FULLTEXT_INDEX = f"""
CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.search_text]
OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{FULLTEXT_ANALYZER}'}}}}
"""
FULLTEXT_INDEX_OPTIONS = (
    "SHOW FULLTEXT INDEXES YIELD name, options WHERE name = $name RETURN options"
)
DROP_FULLTEXT_INDEX = f"DROP INDEX {FULLTEXT_INDEX} IF EXISTS"
AWAIT_INDEX = "CALL db.awaitIndex($name, $seconds)"
INDEX_BUILD_SECONDS = 600

INDEX_INFO_NAME = "chunks"
READ_INDEX_INFO = "MATCH (i:IndexInfo {name: $name}) RETURN i"
WRITE_INDEX_INFO = """
MERGE (i:IndexInfo {name: $name})
SET i.model = $model, i.revision = $revision, i.dimensions = $dimensions,
    i.text_template = $text_template
"""
VECTOR_INDEX_OPTIONS = "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN options"

ARTICLES_WITH_INDEX_STATE = """
MATCH (a:Article)
RETURN a, a.indexed_hash AS indexed_hash ORDER BY a.arxiv_id
"""

# A chunk is immutable: if its key exists (same article, position and text) nothing changes.
UPSERT_CHUNKS = """
UNWIND $rows AS row
MATCH (a:Article {arxiv_id: row.arxiv_id})
MERGE (c:Chunk {key: row.key})
ON CREATE SET c.arxiv_id = row.arxiv_id,
              c.position = row.position,
              c.text = row.text,
              c.search_text = row.search_text,
              c.embedding = row.embedding
WITH c, a
MERGE (c)-[:PART_OF]->(a)
"""

# `articles` rows: {arxiv_id, hash, keys}, with `keys` the keys of the chunks the article owns.
# `indexed_keys` is what the article contributes to the next version.
MARK_INDEXED = """
UNWIND $articles AS art
MATCH (a:Article {arxiv_id: art.arxiv_id})
SET a.indexed_hash = art.hash, a.indexed_keys = art.keys
"""

VERSION_CONSTRAINT = (
    "CREATE CONSTRAINT version_id IF NOT EXISTS FOR (v:Version) REQUIRE v.id IS UNIQUE"
)
CURRENT_CONSTRAINT = (
    "CREATE CONSTRAINT current_name IF NOT EXISTS FOR (c:Current) REQUIRE c.name IS UNIQUE"
)
CURRENT_NAME = "index"

# Articles indexed before versions existed have no `indexed_keys`: they own all their chunks.
ADOPT_LEGACY_CHUNKS = """
MATCH (a:Article) WHERE a.indexed_hash IS NOT NULL AND a.indexed_keys IS NULL
OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(a)
WITH a, [key IN collect(c.key) WHERE key IS NOT NULL] AS keys
SET a.indexed_keys = keys
"""

CREATE_VERSION = """
OPTIONAL MATCH (:Current {name: $name})-[:POINTS_TO]->(previous:Version)
CALL { MATCH (v:Version) RETURN coalesce(max(v.id), 0) + 1 AS next }
CREATE (version:Version {id: next, created_at: datetime(), chunk_count: 0})
FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END |
         CREATE (version)-[:PREVIOUS]->(previous))
RETURN version.id AS id
"""

FILL_MANIFEST = """
MATCH (version:Version {id: $id})
MATCH (a:Article) WHERE a.indexed_keys IS NOT NULL
UNWIND a.indexed_keys AS key
MATCH (c:Chunk {key: key})
MERGE (version)-[:CONTAINS]->(c)
"""

COUNT_MANIFEST = """
MATCH (version:Version {id: $id})
OPTIONAL MATCH (version)-[:CONTAINS]->(c:Chunk)
WITH version, count(c) AS n
SET version.chunk_count = n
RETURN version.id AS id, version.created_at AS created_at, n AS chunk_count
"""

# Moving the pointer is a delete and a create inside the publishing transaction.
POINT_TO_VERSION = """
MERGE (current:Current {name: $name})
WITH current
OPTIONAL MATCH (current)-[old:POINTS_TO]->(:Version)
DELETE old
WITH DISTINCT current
MATCH (version:Version {id: $id})
CREATE (current)-[:POINTS_TO]->(version)
"""

CURRENT_VERSION = """
MATCH (:Current {name: $name})-[:POINTS_TO]->(v:Version)
RETURN v.id AS id, v.created_at AS created_at, v.chunk_count AS chunk_count
"""

PREVIOUS_VERSION = """
MATCH (:Current {name: $name})-[:POINTS_TO]->(:Version)-[:PREVIOUS]->(v:Version)
RETURN v.id AS id, v.created_at AS created_at, v.chunk_count AS chunk_count
"""

# After a rollback each article contributes to the next version the chunks that the version
# rolled back to contains (none if it holds none of the article's).
RESTORE_ARTICLE_KEYS = """
MATCH (a:Article) WHERE a.indexed_keys IS NOT NULL
OPTIONAL MATCH (:Version {id: $id})-[:CONTAINS]->(c:Chunk)-[:PART_OF]->(a)
WITH a, [key IN collect(c.key) WHERE key IS NOT NULL] AS keys
SET a.indexed_keys = keys
"""

# Searches ask the index for more candidates than needed, then keep those of the current
# version. Without any current version (an index built before versions existed) nothing is
# filtered.
FILTER_OVERSAMPLE = 5

READABLE_ONLY = """
OPTIONAL MATCH (:Current {name: $name})-[:POINTS_TO]->(version:Version)
WITH node, score, version
WHERE version IS NULL OR EXISTS { (version)-[:CONTAINS]->(node) }
MATCH (node)-[:PART_OF]->(a:Article)
RETURN a.arxiv_id AS arxiv_id, a.title AS title, node.position AS position,
       node.text AS text, a.abstract AS abstract, score
ORDER BY score DESC
LIMIT $k
"""

VECTOR_SEARCH = (
    "CALL db.index.vector.queryNodes($index, $candidates, $vector) YIELD node, score"
    + READABLE_ONLY
)

KEYWORD_SEARCH = (
    "CALL db.index.fulltext.queryNodes($index, $lucene, {limit: $candidates}) YIELD node, score"
    + READABLE_ONLY
)

# Characters with a meaning in Lucene query syntax; `&` and `|` cover `&&` and `||`.
_LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/')


def lucene_query(text: str) -> str:
    """Turn user text into a Lucene query that matches its words, with no operators.

    Special characters are escaped, and the text is lowercased so that AND, OR, NOT and TO
    are plain words (the default analyzer lowercases the indexed text anyway).
    """
    escaped = "".join(f"\\{ch}" if ch in _LUCENE_SPECIAL else ch for ch in text.lower())
    return " ".join(escaped.split())


def _hit(record: Any) -> SearchHit:
    return SearchHit(
        arxiv_id=record["arxiv_id"],
        title=record["title"],
        position=record["position"],
        text=record["text"],
        score=float(record["score"]),
        abstract=record["abstract"] or "",
    )


def _version(record: Any) -> VersionInfo:
    return VersionInfo(
        id=int(record["id"]),
        created_at=record["created_at"].to_native(),
        chunk_count=int(record["chunk_count"]),
    )


def _article(node: Any) -> Article:
    return Article(
        arxiv_id=node["arxiv_id"],
        title=node["title"],
        abstract=node["abstract"],
        published=node["published"].to_native(),
        categories=tuple(node.get("categories") or ()),
        author_names=tuple(node.get("author_names") or ()),
    )


def _article_row(article: Article, hash_: str) -> dict[str, Any]:
    return {
        "arxiv_id": article.arxiv_id,
        "title": article.title,
        "abstract": article.abstract,
        "published": article.published.isoformat(),
        "categories": list(article.categories),
        "author_names": list(article.author_names),
        "content_hash": hash_,
    }


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(rows)
    while chunk := list(islice(iterator, size)):
        yield chunk


class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        with self._driver.session() as session:
            for statement in SCHEMA:
                session.run(statement).consume()

    def upsert_articles(self, articles: Sequence[Article]) -> ArticleChanges:
        latest = {a.arxiv_id: a for a in articles}  # an id repeated in the batch counts once
        with self._driver.session() as session:
            stored = {
                str(r["id"]): (r["hash"], content_hash(r["title"], r["abstract"]))
                for r in session.run(STORED_ARTICLE_HASHES, ids=list(latest))
            }
        changes = ArticleChanges()
        rows: list[dict[str, Any]] = []
        for a in latest.values():
            current = article_hash(a)
            if a.arxiv_id not in stored:
                changes += ArticleChanges(new=1)
                rows.append(_article_row(a, current))
                continue
            recorded, recomputed = stored[a.arxiv_id]
            # An article stored before hashes existed is compared through its stored text.
            if (recorded or recomputed) != current:
                changes += ArticleChanges(modified=1, modified_ids=(a.arxiv_id,))
                rows.append(_article_row(a, current))
            else:
                changes += ArticleChanges(unchanged=1)
                if recorded is None:
                    rows.append(_article_row(a, current))  # same text: only record its hash
        self._write(rows, UPSERT_ARTICLES)
        return changes

    def upsert_works(self, records: Sequence[WorkRecord]) -> None:
        rows = [
            {
                "arxiv_id": r.arxiv_id,
                "openalex_id": r.openalex_id,
                "authors": [
                    {"key": a.key, "name": a.name, "position": i} for i, a in enumerate(r.authors)
                ],
            }
            for r in records
        ]
        self._write(rows, UPSERT_WORKS, PRUNE_AUTHORS)

    def upsert_citations(self, records: Sequence[ReferenceRecord]) -> None:
        rows = [{"arxiv_id": r.arxiv_id, "cited": list(r.cited_arxiv_ids)} for r in records]
        self._write(rows, PRUNE_CITATIONS, UPSERT_CITATIONS)

    def article_ids(self) -> list[str]:
        with self._driver.session() as session:
            result = session.run("MATCH (a:Article) RETURN a.arxiv_id AS id ORDER BY id")
            return [str(record["id"]) for record in result]

    def counts(self) -> GraphCounts:
        query = """
        CALL () { MATCH (a:Article) RETURN count(a) AS articles }
        CALL () { MATCH (p:Author) RETURN count(p) AS authors }
        CALL () { MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) AS authored_by }
        CALL () { MATCH ()-[r:CITES]->() RETURN count(r) AS cites }
        RETURN articles, authors, authored_by, cites
        """
        with self._driver.session() as session:
            record = session.run(query).single()
        if record is None:
            raise RuntimeError("count query returned no row")
        return GraphCounts(
            articles=record["articles"],
            authors=record["authors"],
            authored_by=record["authored_by"],
            cites=record["cites"],
        )

    def ensure_chunk_schema(self, dimensions: int) -> None:
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError(f"dimensions must be a positive integer, got {dimensions!r}")
        with self._driver.session() as session:
            session.run(CHUNK_KEY_CONSTRAINT).consume()
            session.run(INDEX_INFO_CONSTRAINT).consume()
            session.run(VERSION_CONSTRAINT).consume()
            session.run(CURRENT_CONSTRAINT).consume()
            session.run(create_vector_index(dimensions)).consume()
            self._ensure_fulltext_index(session)
            record = session.run(VECTOR_INDEX_OPTIONS, name=VECTOR_INDEX).single()
        if record is None:
            raise RuntimeError(f"vector index {VECTOR_INDEX} was not created")
        existing = record["options"]["indexConfig"]["vector.dimensions"]
        if existing != dimensions:
            raise ValueError(
                f"vector index {VECTOR_INDEX} has {existing} dimensions, the embedder has "
                f"{dimensions}: drop the index and the chunks, then index again"
            )

    def read_index_info(self) -> IndexInfo | None:
        """What the stored vectors were built with, or None if nothing is recorded yet."""
        with self._driver.session() as session:
            record = session.run(READ_INDEX_INFO, name=INDEX_INFO_NAME).single()
        if record is None:
            return None
        node = record["i"]
        return IndexInfo(
            model=node["model"],
            revision=node["revision"],
            dimensions=node["dimensions"],
            text_template=node["text_template"],
        )

    def ensure_index_info(self, info: IndexInfo) -> None:
        """Record the fingerprint if none exists; refuse any other one.

        A database indexed before this check existed has no fingerprint: the first call
        adopts the current one, so run it only with the model that built the vectors.
        """
        stored = self.read_index_info()
        if stored is None:
            with self._driver.session() as session:
                session.run(
                    WRITE_INDEX_INFO,
                    name=INDEX_INFO_NAME,
                    model=info.model,
                    revision=info.revision,
                    dimensions=info.dimensions,
                    text_template=info.text_template,
                ).consume()
            return
        self.check_index_info(info)

    def check_index_info(self, info: IndexInfo) -> None:
        """Raise unless the stored vectors were built with `info` (use before searching)."""
        stored = self.read_index_info()
        if stored is None:
            raise ValueError("no index fingerprint is recorded: run graphrag.cli.index once")
        differences = stored.differences(info)
        if differences:
            raise ValueError(
                "the stored vectors were built differently: "
                + "; ".join(differences)
                + ". Drop the chunks and index again, or use the original model"
            )

    @staticmethod
    def _ensure_fulltext_index(session: neo4j.Session) -> None:
        """Create the keyword index, rebuilding it if it uses another analyzer.

        The index only copies `search_text`, so dropping it loses nothing.
        """
        record = session.run(FULLTEXT_INDEX_OPTIONS, name=FULLTEXT_INDEX).single()
        if record is not None:
            current = record["options"]["indexConfig"].get("fulltext.analyzer")
            if current != FULLTEXT_ANALYZER:
                session.run(DROP_FULLTEXT_INDEX).consume()
        session.run(CREATE_FULLTEXT_INDEX).consume()
        session.run(AWAIT_INDEX, name=FULLTEXT_INDEX, seconds=INDEX_BUILD_SECONDS).consume()

    def all_articles(self) -> list[Article]:
        with self._driver.session() as session:
            result = session.run("MATCH (a:Article) RETURN a ORDER BY a.arxiv_id")
            return [_article(record["a"]) for record in result]

    def articles_with_index_state(self) -> list[ArticleIndexState]:
        with self._driver.session() as session:
            return [
                ArticleIndexState(_article(record["a"]), record["indexed_hash"])
                for record in session.run(ARTICLES_WITH_INDEX_STATE)
            ]

    def upsert_chunks(
        self, chunks: Sequence[IndexedChunk], article_hashes: Mapping[str, str] | None = None
    ) -> None:
        rows = [
            {
                "key": c.key,
                "arxiv_id": c.chunk.arxiv_id,
                "position": c.chunk.position,
                "text": c.chunk.text,
                "search_text": c.search_text,
                "embedding": list(c.embedding),
            }
            for c in chunks
        ]
        owners = article_hashes or {}
        if not rows and not owners:
            return
        kept: dict[str, list[str]] = {article_id: [] for article_id in owners}
        for c in chunks:
            kept.setdefault(c.chunk.arxiv_id, []).append(c.key)
        articles = [
            {"arxiv_id": article_id, "hash": owners[article_id], "keys": kept[article_id]}
            for article_id in owners
        ]

        def write(tx: neo4j.ManagedTransaction) -> None:
            if rows:
                tx.run(UPSERT_CHUNKS, rows=rows).consume()
            if articles:
                tx.run(MARK_INDEXED, articles=articles).consume()

        with self._driver.session() as session:
            # one transaction: an article is never half indexed
            session.execute_write(write)

    def publish_version(self) -> VersionInfo:
        def write(tx: neo4j.ManagedTransaction) -> VersionInfo:
            tx.run(ADOPT_LEGACY_CHUNKS).consume()
            version_id = tx.run(CREATE_VERSION, name=CURRENT_NAME).single(strict=True)["id"]
            tx.run(FILL_MANIFEST, id=version_id).consume()
            counted = tx.run(COUNT_MANIFEST, id=version_id).single(strict=True)
            tx.run(POINT_TO_VERSION, name=CURRENT_NAME, id=version_id).consume()
            return _version(counted)

        with self._driver.session() as session:
            # one transaction: the manifest and the pointer change together or not at all
            return session.execute_write(write)

    def current_version(self) -> VersionInfo | None:
        with self._driver.session() as session:
            record = session.run(CURRENT_VERSION, name=CURRENT_NAME).single()
        return None if record is None else _version(record)

    def rollback(self) -> VersionInfo:
        def write(tx: neo4j.ManagedTransaction) -> VersionInfo:
            previous = tx.run(PREVIOUS_VERSION, name=CURRENT_NAME).single()
            if previous is None:
                raise LookupError("no earlier version to roll back to")
            tx.run(POINT_TO_VERSION, name=CURRENT_NAME, id=previous["id"]).consume()
            tx.run(RESTORE_ARTICLE_KEYS, id=previous["id"]).consume()
            return _version(previous)

        with self._driver.session() as session:
            return session.execute_write(write)

    def vector_search(self, vector: Sequence[float], k: int) -> list[SearchHit]:
        with self._driver.session() as session:
            result = session.run(
                VECTOR_SEARCH,
                index=VECTOR_INDEX,
                name=CURRENT_NAME,
                candidates=k * FILTER_OVERSAMPLE,
                k=k,
                vector=list(vector),
            )
            return [_hit(record) for record in result]

    def keyword_search(self, text: str, k: int) -> list[SearchHit]:
        if not any(ch.isalnum() for ch in text):
            return []  # nothing to match, and Lucene rejects an empty query
        query = lucene_query(text)
        with self._driver.session() as session:
            result = session.run(
                KEYWORD_SEARCH,
                index=FULLTEXT_INDEX,
                name=CURRENT_NAME,
                candidates=k * FILTER_OVERSAMPLE,
                lucene=query,
                k=k,
            )
            return [_hit(record) for record in result]

    def chunk_count(self) -> int:
        with self._driver.session() as session:
            record = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()
        return 0 if record is None else int(record["n"])

    def delete_article(self, arxiv_id: str) -> bool:
        def delete(tx: neo4j.ManagedTransaction) -> bool:
            found = tx.run(AUTHORS_OF_ARTICLE, id=arxiv_id).single()
            if found is None or not found["n"]:
                return False
            tx.run(DELETE_ARTICLE, id=arxiv_id).consume()
            tx.run(DELETE_ORPHAN_AUTHORS, keys=found["keys"]).consume()
            return True

        with self._driver.session() as session:
            return session.execute_write(delete)

    def _write(self, rows: Sequence[dict[str, Any]], *queries: str) -> None:
        """Run `queries` in order over `rows`, one transaction per batch of rows."""

        def run_chunk(tx: neo4j.ManagedTransaction, chunk: list[dict[str, Any]]) -> None:
            for query in queries:
                tx.run(query, rows=chunk).consume()

        with self._driver.session() as session:
            for chunk in _chunks(rows, WRITE_BATCH):
                session.execute_write(run_chunk, chunk)
