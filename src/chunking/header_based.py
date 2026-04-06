"""Chunking po nagłówkach — naturalna struktura artykułu SEO.

Dzieli artykuł po nagłówkach H2/H3 (BLUF i sekcje tematyczne).
Sekcje dłuższe niż header_max_chunk_size są dodatkowo dzielone
za pomocą RecursiveCharacterTextSplitter.
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config.settings import ChunkingConfig, get_chunking_config
from src.chunking.base import Chunk, ChunkerBase


class HeaderChunker(ChunkerBase):
    """Podział tekstu po nagłówkach H2/H3 z sub-splitem długich sekcji."""

    def __init__(self, config: ChunkingConfig | None = None):
        cfg = config or get_chunking_config()
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=cfg.header_split_on,
        )
        self._max_chunk_size = cfg.header_max_chunk_size
        self._sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.header_max_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "],
        )

    @property
    def name(self) -> str:
        return "header"

    def chunk(self, markdown_text: str, metadata: dict | None = None) -> list[Chunk]:
        """Podziel tekst po nagłówkach H2/H3.

        Sekcje dłuższe niż max_chunk_size są dodatkow dzielone.

        Args:
            markdown_text: Tekst artykułu w markdown.
            metadata: Bazowe metadane.

        Returns:
            Lista chunków z metadanymi (h2, h3, method).
        """
        base_meta = metadata or {}
        slug = base_meta.get("slug", "unknown")

        # Podział po nagłówkach
        dokumenty = self._header_splitter.split_text(markdown_text)

        chunks = []
        indeks = 0

        for doc in dokumenty:
            tekst = doc.page_content
            header_meta = doc.metadata  # np. {'h2': 'Sekcja', 'h3': 'Podsekcja'}

            # Sub-split jeśli sekcja za długa
            if len(tekst) > self._max_chunk_size:
                sub_fragmenty = self._sub_splitter.split_text(tekst)
            else:
                sub_fragmenty = [tekst]

            for sub_tekst in sub_fragmenty:
                chunk_meta = {
                    **base_meta,
                    **header_meta,
                    "method": self.name,
                    "chunk_index": indeks,
                    "char_count": len(sub_tekst),
                }
                chunks.append(
                    Chunk(
                        text=sub_tekst,
                        metadata=chunk_meta,
                        chunk_id=self._buduj_chunk_id(slug, indeks),
                    )
                )
                indeks += 1

        return chunks
