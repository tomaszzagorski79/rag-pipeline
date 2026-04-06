"""Hybrid Search — łączenie BM25 (sparse) z wyszukiwaniem wektorowym (dense).

Używa Qdrant Query API: prefetch z obu źródeł → Reciprocal Rank Fusion (RRF).
"""

from dataclasses import dataclass

from qdrant_client import models

from src.embeddings.jina_embed import JinaDenseEmbedder
from src.embeddings.sparse_embed import SparseBM25Embedder
from src.vectorstore.qdrant_store import QdrantStore


@dataclass
class SearchResult:
    """Wynik wyszukiwania z tekstem, score i metadanymi."""

    text: str
    score: float
    metadata: dict
    chunk_id: str


class HybridRetriever:
    """Retriever łączący dense (Jina) i sparse (BM25) search przez RRF.

    Architektura:
    1. Embed query (dense + sparse)
    2. Prefetch top-N z obu źródeł
    3. RRF fusion → top-K wyników
    """

    def __init__(
        self,
        store: QdrantStore | None = None,
        dense_embedder: JinaDenseEmbedder | None = None,
        sparse_embedder: SparseBM25Embedder | None = None,
    ):
        self.store = store or QdrantStore()
        self.dense = dense_embedder or JinaDenseEmbedder()
        self.sparse = sparse_embedder or SparseBM25Embedder()

    def search(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        prefetch_limit: int = 20,
    ) -> list[SearchResult]:
        """Wyszukaj dokumenty hybrid searchem (dense + sparse → RRF).

        Args:
            query: Zapytanie użytkownika.
            collection_name: Nazwa kolekcji Qdrant.
            limit: Liczba zwracanych wyników.
            prefetch_limit: Liczba kandydatów z każdego źródła.

        Returns:
            Lista SearchResult posortowana po score (malejąco).
        """
        # 1. Embed query obiema metodami
        dense_vector = self.dense.embed_query(query)
        sparse_vector = self.sparse.embed_query(query)

        # 2. Hybrid search z RRF fusion
        results = self.store.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using="sparse",
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        # 3. Konwersja na SearchResult
        search_results = []
        for point in results.points:
            payload = point.payload or {}
            search_results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=point.score,
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ("text", "chunk_id")
                    },
                    chunk_id=payload.get("chunk_id", ""),
                )
            )

        return search_results

    def search_dense_only(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Wyszukaj tylko wektorowo (bez BM25) — do porównań."""
        dense_vector = self.dense.embed_query(query)

        results = self.store.client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )

        return [
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
