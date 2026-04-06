"""Moduł embeddingów Jina v5 text-small.

Zawiera:
- JinaDenseEmbedder: bezpośredni klient API Jina do embeddingów dense
- JinaEmbeddingsLangChain: adapter LangChain Embeddings (dla SemanticChunker i RAGAS)
"""

import time

import httpx
from langchain_core.embeddings import Embeddings

from config.settings import JinaConfig, get_jina_config


class JinaDenseEmbedder:
    """Klient API Jina do generowania embeddingów dense.

    Obsługuje task-specific LoRA adapters (retrieval.passage / retrieval.query)
    i automatyczny batching.
    """

    def __init__(self, config: JinaConfig | None = None):
        self.config = config or get_jina_config()
        self._client = httpx.Client(timeout=60.0)

    def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        """Wysyła batch tekstów do API i zwraca embeddingi.

        Args:
            texts: Lista tekstów do embeddingu.
            task: Typ zadania ('retrieval.passage' lub 'retrieval.query').

        Returns:
            Lista wektorów embeddingów.
        """
        response = self._client.post(
            self.config.embed_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "input": texts,
                "task": task,
                "dimensions": self.config.dimensions,
            },
        )
        response.raise_for_status()
        data = response.json()
        # Sortuj po indeksie (API może zwrócić w innej kolejności)
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generuj embeddingi dla dokumentów (chunków).

        Automatycznie dzieli na batche po max_batch_size.

        Args:
            texts: Lista tekstów dokumentów.

        Returns:
            Lista wektorów embeddingów (1024d).
        """
        all_embeddings = []
        batch_size = self.config.max_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._embed_batch(batch, task="retrieval.passage")
            all_embeddings.extend(embeddings)

            # Rate limiting przy wielu batchach
            if i + batch_size < len(texts):
                time.sleep(0.5)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generuj embedding dla zapytania użytkownika.

        Używa task='retrieval.query' (inny adapter LoRA niż dla dokumentów).

        Args:
            text: Tekst zapytania.

        Returns:
            Wektor embeddingu (1024d).
        """
        result = self._embed_batch([text], task="retrieval.query")
        return result[0]

    def close(self):
        """Zamknij klienta HTTP."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class JinaEmbeddingsLangChain(Embeddings):
    """Adapter LangChain Embeddings dla Jina v5.

    Potrzebny dla:
    - SemanticChunker (langchain_experimental)
    - RAGAS (ewaluacja)
    """

    def __init__(self, config: JinaConfig | None = None):
        self._embedder = JinaDenseEmbedder(config)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeddingi dla listy dokumentów."""
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embedding dla zapytania."""
        return self._embedder.embed_query(text)
