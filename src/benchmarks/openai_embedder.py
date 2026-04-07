"""Wrapper dla Google Gemini Embedding API."""

import os


class GeminiEmbedder:
    """Embedder oparty na Google Gemini Embedding API."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
    ):
        self._model = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self._client = None

    def _load(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed listy tekstów (jako dokumenty/passage)."""
        self._load()
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed pojedynczego zapytania."""
        self._load()
        result = self._client.models.embed_content(
            model=self._model,
            contents=[text],
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return result.embeddings[0].values

    @property
    def name(self) -> str:
        return self._model
