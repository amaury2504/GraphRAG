import neo4j
import psycopg


class Neo4jProbe:
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._auth = (user, password)

    def check(self) -> None:
        with neo4j.GraphDatabase.driver(self._uri, auth=self._auth) as driver:
            driver.verify_connectivity()

    def unindexed_articles(self) -> int:
        """Articles never indexed or changed since: `cli.index` has work to do."""
        query = (
            "MATCH (a:Article) WHERE a.indexed_hash IS NULL "
            "OR (a.content_hash IS NOT NULL AND a.indexed_hash <> a.content_hash) "
            "RETURN count(a) AS n"
        )
        with neo4j.GraphDatabase.driver(self._uri, auth=self._auth) as driver:
            records, _, _ = driver.execute_query(query)
        return int(records[0]["n"])


class PostgresProbe:
    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def check(self) -> None:
        with psycopg.connect(self._dsn, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
