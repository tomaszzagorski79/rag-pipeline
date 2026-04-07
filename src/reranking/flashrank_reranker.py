"""Re-ranking wyników wyszukiwania za pomocą cross-encodera (FlashRank).

Cross-encoder analizuje parę (pytanie, fragment) razem — dokładniejszy niż
bi-encoder (embedding), ale wolniejszy. Stosowany po retrieval na top-N kandydatach.
"""

from flashrank import Ranker, RerankRequest

from src.retrieval.hybrid_search import SearchResult


class FlashRankReranker:
    """Re-ranker oparty na FlashRank (lokalny, bez API)."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """Inicjalizacja re-rankera.

        Args:
            model_name: Nazwa modelu FlashRank. Domyślny jest angielski.
                        Dla lepszego wsparcia polskiego: 'ms-marco-MultiBERT-L-12'.
        """
        self._ranker = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-rankuj wyniki wyszukiwania cross-encoderem.

        Args:
            query: Zapytanie użytkownika.
            results: Lista SearchResult z hybrid search.
            top_k: Liczba wyników po re-rankingu.

        Returns:
            Lista SearchResult posortowana po nowym score, z original_score w metadata.
        """
        if not results:
            return []

        # Przygotuj dane dla FlashRank
        passages = [
            {"id": i, "text": r.text, "meta": {"original": r}}
            for i, r in enumerate(results)
        ]

        # Re-rank
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = self._ranker.rerank(rerank_request)

        # Konwersja na SearchResult z zachowaniem oryginalnego score
        reranked_results = []
        for item in reranked[:top_k]:
            original: SearchResult = item["meta"]["original"]
            reranked_results.append(
                SearchResult(
                    text=original.text,
                    score=item["score"],
                    metadata={
                        **original.metadata,
                        "original_score": original.score,
                        "rerank_model": self._ranker.model_name if hasattr(self._ranker, 'model_name') else "flashrank",
                    },
                    chunk_id=original.chunk_id,
                )
            )

        return reranked_results
