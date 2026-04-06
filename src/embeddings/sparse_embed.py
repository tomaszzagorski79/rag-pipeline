"""Moduł embeddingów sparse (BM25) przez FastEmbed.

Generuje sparse vectors do hybrid search w Qdrant.
FastEmbed model 'Qdrant/bm25' automatycznie tokenizuje tekst.
"""

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector


class SparseBM25Embedder:
    """Klient FastEmbed do generowania sparse vectors (BM25).

    Qdrant automatycznie stosuje IDF modifier po stronie serwera,
    więc tutaj generujemy tylko surowe częstotliwości tokenów.
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        """Inicjalizacja modelu sparse.

        Args:
            model_name: Nazwa modelu FastEmbed (domyślnie 'Qdrant/bm25').
        """
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Generuj sparse vectors dla listy dokumentów.

        Args:
            texts: Lista tekstów do embeddingu.

        Returns:
            Lista SparseVector (indices + values).
        """
        wyniki = list(self._model.embed(texts))
        return [
            SparseVector(
                indices=w.indices.tolist(),
                values=w.values.tolist(),
            )
            for w in wyniki
        ]

    def embed_query(self, text: str) -> SparseVector:
        """Generuj sparse vector dla zapytania.

        Args:
            text: Tekst zapytania.

        Returns:
            SparseVector z indeksami i wartościami.
        """
        wyniki = list(self._model.embed([text]))
        w = wyniki[0]
        return SparseVector(
            indices=w.indices.tolist(),
            values=w.values.tolist(),
        )
