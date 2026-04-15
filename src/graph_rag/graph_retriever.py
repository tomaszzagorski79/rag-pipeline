"""Graph Retriever — wyszukiwanie w grafie wiedzy.

Rozbija zapytanie na encje (Claude), szuka ich w grafie, zwraca kontekst
z linkowanych chunków i atrybutów.
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config
from src.graph_rag.graph_store import Neo4jStore

EXTRACT_ENTITIES_PROMPT = """Wyciągnij z pytania nazwy kluczowych encji (marki, produkty, kraje, pojęcia).

Pytanie: {query}

Zwróć JSON array z listą encji (1-5, krótkie nazwy).
Przykład: ["VAT", "Niemcy"], ["Amazon", "logistyka"], ["cross-border"]

Zwróć TYLKO JSON array."""


@dataclass
class GraphContext:
    """Kontekst z grafu dla jednej encji."""

    entity: str
    mentions: list[dict] = field(default_factory=list)  # chunk_id, text, attrs


@dataclass
class GraphRAGResult:
    """Wynik Graph RAG."""

    query: str
    extracted_entities: list[str] = field(default_factory=list)
    graph_contexts: list[GraphContext] = field(default_factory=list)
    answer: str = ""


class GraphRetriever:
    """Retriever wyszukujący w grafie wiedzy Neo4j."""

    def __init__(self, store: Neo4jStore | None = None):
        self._store = store or Neo4jStore()
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def _extract_entities(self, query: str) -> list[str]:
        """Wyciągnij encje z zapytania."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[
                {"role": "user", "content": EXTRACT_ENTITIES_PROMPT.format(query=query)},
            ],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            entities = json.loads(raw)
            if isinstance(entities, list):
                return [str(e) for e in entities]
        except json.JSONDecodeError:
            pass
        return []

    def search(self, query: str, limit_per_entity: int = 3) -> GraphRAGResult:
        """Wyszukaj w grafie na podstawie encji z zapytania.

        Args:
            query: Pytanie użytkownika.
            limit_per_entity: Ile chunków per encja.

        Returns:
            GraphRAGResult ze wszystkimi kontekstami.
        """
        result = GraphRAGResult(query=query)

        # 1. Wyciągnij encje
        entities = self._extract_entities(query)
        result.extracted_entities = entities

        # 2. Szukaj w grafie dla każdej encji
        for entity in entities:
            mentions = self._store.search_by_entity(entity, limit=limit_per_entity)
            if mentions:
                result.graph_contexts.append(
                    GraphContext(entity=entity, mentions=mentions)
                )

        return result

    def close(self):
        self._client.close()
