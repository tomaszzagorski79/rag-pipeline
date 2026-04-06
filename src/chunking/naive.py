"""Naiwny chunking — podział tekstu co N znaków z overlapem.

Baseline do porównań. Szybki, ale gubi kontekst semantyczny.
Używa RecursiveCharacterTextSplitter z LangChain.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import ChunkingConfig, get_chunking_config
from src.chunking.base import Chunk, ChunkerBase


class NaiveChunker(ChunkerBase):
    """Podział tekstu na fragmenty o stałej długości z overlapem."""

    def __init__(self, config: ChunkingConfig | None = None):
        cfg = config or get_chunking_config()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.naive_chunk_size,
            chunk_overlap=cfg.naive_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    @property
    def name(self) -> str:
        return "naive"

    def chunk(self, markdown_text: str, metadata: dict | None = None) -> list[Chunk]:
        """Podziel tekst na fragmenty o stałej długości.

        Args:
            markdown_text: Tekst artykułu w markdown.
            metadata: Bazowe metadane (np. slug, source_file).

        Returns:
            Lista chunków z metadanymi (method, chunk_index).
        """
        base_meta = metadata or {}
        slug = base_meta.get("slug", "unknown")

        fragmenty = self._splitter.split_text(markdown_text)

        chunks = []
        for i, tekst in enumerate(fragmenty):
            chunk_meta = {
                **base_meta,
                "method": self.name,
                "chunk_index": i,
                "char_count": len(tekst),
            }
            chunks.append(
                Chunk(
                    text=tekst,
                    metadata=chunk_meta,
                    chunk_id=self._buduj_chunk_id(slug, i),
                )
            )

        return chunks
