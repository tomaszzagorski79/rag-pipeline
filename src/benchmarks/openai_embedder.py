"""Wrapper dla OpenAI text-embedding-3-small."""

import os


class OpenAIEmbedder:
    """Embedder oparty na OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _load(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed listy tekstów."""
        self._load()
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed pojedynczego zapytania."""
        result = self.embed_documents([text])
        return result[0]

    @property
    def name(self) -> str:
        return self._model
