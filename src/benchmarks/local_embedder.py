"""Wrapper dla lokalnych modeli embeddingowych (sentence-transformers)."""

import numpy as np


class LocalSentenceTransformerEmbedder:
    """Embedder oparty na sentence-transformers (lokalne modele HuggingFace)."""

    def __init__(self, model_name: str):
        """Inicjalizacja (lazy load — model ładuje się przy pierwszym użyciu).

        Args:
            model_name: Nazwa modelu HF, np. 'intfloat/multilingual-e5-large'.
        """
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed listy tekstów."""
        self._load()
        # multilingual-e5 wymaga prefixu
        if "e5" in self._model_name.lower():
            texts = [f"passage: {t}" for t in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed pojedynczego zapytania."""
        self._load()
        if "e5" in self._model_name.lower():
            text = f"query: {text}"
        embedding = self._model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

    @property
    def name(self) -> str:
        return self._model_name.split("/")[-1]
