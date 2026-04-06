"""Chunking semantyczny — podział na podstawie zmian tematycznych.

Używa SemanticChunker z LangChain, który analizuje embeddingi zdań
i dzieli tekst tam, gdzie następuje zmiana tematu.
"""

from langchain_experimental.text_splitter import SemanticChunker

from config.settings import ChunkingConfig, get_chunking_config
from src.chunking.base import Chunk, ChunkerBase


class SemanticChunkerWrapper(ChunkerBase):
    """Podział tekstu na podstawie semantycznych zmian tematycznych."""

    def __init__(
        self,
        embeddings,
        config: ChunkingConfig | None = None,
    ):
        """Inicjalizacja SemanticChunker.

        Args:
            embeddings: Obiekt implementujący interfejs LangChain Embeddings
                        (np. JinaEmbeddingsLangChain).
            config: Konfiguracja chunkingu.
        """
        cfg = config or get_chunking_config()
        self._chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=cfg.semantic_breakpoint_type,
            breakpoint_threshold_amount=cfg.semantic_breakpoint_threshold,
        )

    @property
    def name(self) -> str:
        return "semantic"

    def chunk(self, markdown_text: str, metadata: dict | None = None) -> list[Chunk]:
        """Podziel tekst na semantycznie spójne fragmenty.

        Args:
            markdown_text: Tekst artykułu w markdown.
            metadata: Bazowe metadane.

        Returns:
            Lista chunków z metadanymi (method, chunk_index).
        """
        base_meta = metadata or {}
        slug = base_meta.get("slug", "unknown")

        dokumenty = self._chunker.create_documents([markdown_text])

        chunks = []
        for i, doc in enumerate(dokumenty):
            chunk_meta = {
                **base_meta,
                "method": self.name,
                "chunk_index": i,
                "char_count": len(doc.page_content),
            }
            chunks.append(
                Chunk(
                    text=doc.page_content,
                    metadata=chunk_meta,
                    chunk_id=self._buduj_chunk_id(slug, i),
                )
            )

        return chunks
