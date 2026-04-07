"""HyDE — Hypothetical Document Embeddings.

Zamiast szukać dokumentów pasujących do pytania, generuje "idealną odpowiedź"
i szuka dokumentów pasujących do TEJ odpowiedzi. Zmienia optykę retrieval.
"""

from src.embeddings.jina_embed import JinaDenseEmbedder
from src.generation.claude_gen import ClaudeGenerator
from src.retrieval.hybrid_search import SearchResult
from src.vectorstore.qdrant_store import QdrantStore

HYDE_SYSTEM_PROMPT = """Jesteś ekspertem e-commerce i SEO. Na podstawie pytania użytkownika
wygeneruj IDEALNĄ, wyczerpującą odpowiedź w 2-3 akapitach po polsku.

Pisz tak, jakbyś pisał fragment artykułu eksperciego. Podawaj konkrety, liczby, przykłady.
NIE zaznaczaj, że to jest hipoteza — pisz pewnie, jakbyś był autorem artykułu.
"""


class HyDEGenerator:
    """Generator hipotetycznych dokumentów do wyszukiwania."""

    def __init__(
        self,
        generator: ClaudeGenerator | None = None,
        embedder: JinaDenseEmbedder | None = None,
        store: QdrantStore | None = None,
    ):
        self._generator = generator or ClaudeGenerator()
        self._embedder = embedder or JinaDenseEmbedder()
        self._store = store or QdrantStore()

    def generate_hypothesis(self, query: str) -> str:
        """Wygeneruj hipotetyczną idealną odpowiedź na pytanie.

        Args:
            query: Pytanie użytkownika.

        Returns:
            Tekst hipotetycznej odpowiedzi (2-3 akapity).
        """
        return self._generator.generate(
            query=query,
            contexts=[],  # brak kontekstu — czysta generacja
            system_prompt=HYDE_SYSTEM_PROMPT,
        )

    def search_with_hyde(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
    ) -> tuple[str, list[SearchResult]]:
        """Wygeneruj hipotezę → embed → szukaj podobnych dokumentów.

        Args:
            query: Pytanie użytkownika.
            collection_name: Nazwa kolekcji Qdrant.
            limit: Liczba wyników.

        Returns:
            Tuple (hipoteza, wyniki wyszukiwania).
        """
        # 1. Generuj hipotezę
        hypothesis = self.generate_hypothesis(query)

        # 2. Embed hipotezę (jako "passage" — bo wygląda jak dokument)
        hyde_vector = self._embedder.embed_documents([hypothesis])[0]

        # 3. Dense-only search z wektorem hipotezy
        from qdrant_client import models
        results = self._store.client.query_points(
            collection_name=collection_name,
            query=hyde_vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )

        search_results = [
            SearchResult(
                text=(point.payload or {}).get("text", ""),
                score=point.score,
                metadata={
                    k: v
                    for k, v in (point.payload or {}).items()
                    if k not in ("text", "chunk_id")
                },
                chunk_id=(point.payload or {}).get("chunk_id", ""),
            )
            for point in results.points
        ]

        return hypothesis, search_results
