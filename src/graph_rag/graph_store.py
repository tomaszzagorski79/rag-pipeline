"""Graph RAG — klient Neo4j do przechowywania grafu EAV.

Model danych:
- Node :Entity {name}
- Node :Chunk {chunk_id, text, slug}
- Relationship (Entity)-[:HAS_ATTRIBUTE {attribute, value}]->(Entity|Value)
- Relationship (Chunk)-[:MENTIONS]->(Entity)

Wymaga: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD w .env
"""

import os

from neo4j import GraphDatabase


class Neo4jStore:
    """Klient Neo4j dla Graph RAG."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")

        if not self._uri or not self._password:
            raise ValueError(
                "Brak konfiguracji Neo4j. Dodaj NEO4J_URI i NEO4J_PASSWORD do .env. "
                "Użyj Neo4j Aura (free): https://console.neo4j.io/ "
                "lub Dockera: docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j"
            )

        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def verify(self) -> bool:
        """Sprawdź połączenie z Neo4j."""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def clear_graph(self) -> None:
        """Wyczyść cały graf."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def upsert_eav_triples(
        self,
        chunk_id: str,
        chunk_text: str,
        slug: str,
        triples: list[dict],
    ) -> None:
        """Wstaw trójki EAV do grafu z linkiem do chunka.

        Args:
            chunk_id: ID chunka (źródło).
            chunk_text: Treść chunka.
            slug: Slug artykułu.
            triples: Lista dict z kluczami: entity, attribute, value.
        """
        with self._driver.session() as session:
            # Utwórz chunk node
            session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text = $text, c.slug = $slug
                """,
                chunk_id=chunk_id,
                text=chunk_text[:500],
                slug=slug,
            )

            # Utwórz trójki EAV
            for t in triples:
                entity = t.get("entity", "").strip()
                attribute = t.get("attribute", "").strip()
                value = t.get("value", "").strip()

                if not entity or not attribute or not value:
                    continue

                session.run(
                    """
                    MERGE (e:Entity {name: $entity})
                    MERGE (v:Value {text: $value})
                    MERGE (e)-[r:HAS_ATTRIBUTE {name: $attribute}]->(v)
                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    entity=entity,
                    attribute=attribute,
                    value=value,
                    chunk_id=chunk_id,
                )

    def search_by_entity(self, entity_query: str, limit: int = 5) -> list[dict]:
        """Znajdź chunki wspominające encję (fuzzy match na name).

        Args:
            entity_query: Szukany tekst encji.
            limit: Liczba wyników.

        Returns:
            Lista dict z chunk_id, text, entities, attributes.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                OPTIONAL MATCH (e)-[r:HAS_ATTRIBUTE]->(v:Value)
                WITH c, e, collect(DISTINCT {attribute: r.name, value: v.text}) as attrs
                RETURN c.chunk_id as chunk_id, c.text as text, c.slug as slug,
                       e.name as entity, attrs
                LIMIT $limit
                """,
                query=entity_query,
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_stats(self) -> dict:
        """Statystyki grafu."""
        with self._driver.session() as session:
            result = session.run("""
                MATCH (e:Entity) WITH count(e) as entities
                MATCH (c:Chunk) WITH entities, count(c) as chunks
                MATCH ()-[r:HAS_ATTRIBUTE]->() WITH entities, chunks, count(r) as attributes
                MATCH ()-[m:MENTIONS]->() RETURN entities, chunks, attributes, count(m) as mentions
            """)
            record = result.single()
            return dict(record) if record else {"entities": 0, "chunks": 0, "attributes": 0, "mentions": 0}

    def get_top_entities(self, limit: int = 20) -> list[dict]:
        """Top encje po liczbie wzmianek."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WITH e, count(c) as mentions
                ORDER BY mentions DESC
                LIMIT $limit
                OPTIONAL MATCH (e)-[r:HAS_ATTRIBUTE]->(v:Value)
                RETURN e.name as entity, mentions, count(r) as num_attributes
                """,
                limit=limit,
            )
            return [dict(record) for record in result]

    def close(self):
        self._driver.close()
