"""Sentence-level chunking — każde zdanie to osobny chunk.

Najdrobniejsza granulacja. Dobre do FAQ i krótkich odpowiedzi.
"""

import re

from src.chunking.base import Chunk, ChunkerBase


class SentenceChunker(ChunkerBase):
    """Podział tekstu na pojedyncze zdania."""

    @property
    def name(self) -> str:
        return "sentence"

    def chunk(self, markdown_text: str, metadata: dict | None = None) -> list[Chunk]:
        """Podziel tekst na zdania.

        Args:
            markdown_text: Tekst artykułu.
            metadata: Bazowe metadane.

        Returns:
            Lista Chunk — każdy = jedno zdanie.
        """
        base_meta = metadata or {}
        slug = base_meta.get("slug", "unknown")

        # Usuń nagłówki markdown i puste linie
        clean = re.sub(r"^#{1,6}\s+.*$", "", markdown_text, flags=re.MULTILINE)
        clean = re.sub(r"\n{3,}", "\n\n", clean)

        # Split na zdania (po . ? ! z uwzględnieniem skrótów)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZŻŹĆĄŚĘŁÓŃ])', clean)

        chunks = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 15:  # pomiń za krótkie fragmenty
                continue

            chunk_meta = {
                **base_meta,
                "method": self.name,
                "chunk_index": i,
                "char_count": len(sent),
            }
            chunks.append(Chunk(
                text=sent,
                metadata=chunk_meta,
                chunk_id=self._buduj_chunk_id(slug, i),
            ))

        return chunks
