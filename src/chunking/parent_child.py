"""Parent-child chunking (small-to-big).

Małe chunki do retrieval (precyzyjne szukanie),
ale do LLM trafia CAŁY parent (pełny kontekst sekcji H2).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import ChunkingConfig, get_chunking_config
from src.chunking.base import Chunk, ChunkerBase


class ParentChildChunker(ChunkerBase):
    """Small-to-big: małe chunki do szukania, duże do generowania."""

    def __init__(self, config: ChunkingConfig | None = None):
        cfg = config or get_chunking_config()
        self._child_size = 300  # małe chunki do retrieval
        self._child_overlap = 50

    @property
    def name(self) -> str:
        return "parent_child"

    def chunk(self, markdown_text: str, metadata: dict | None = None) -> list[Chunk]:
        """Podziel na małe chunki (child) z referencją do dużego parenta (sekcja H2).

        Każdy chunk ma w metadata klucz 'parent_text' z pełnym tekstem sekcji.

        Args:
            markdown_text: Tekst artykułu.
            metadata: Bazowe metadane.

        Returns:
            Lista małych Chunków, każdy z parent_text w metadata.
        """
        base_meta = metadata or {}
        slug = base_meta.get("slug", "unknown")

        # Wyciągnij sekcje H2 jako parenty
        sections = self._extract_sections(markdown_text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_size,
            chunk_overlap=self._child_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        chunks = []
        idx = 0

        for section in sections:
            parent_text = section["content"]
            h2 = section["header"]

            # Podziel parent na małe children
            children = splitter.split_text(parent_text)

            for child_text in children:
                chunk_meta = {
                    **base_meta,
                    "method": self.name,
                    "chunk_index": idx,
                    "char_count": len(child_text),
                    "h2": h2,
                    "parent_text": parent_text,  # pełny tekst parenta
                    "parent_length": len(parent_text),
                }
                chunks.append(Chunk(
                    text=child_text,
                    metadata=chunk_meta,
                    chunk_id=self._buduj_chunk_id(slug, idx),
                ))
                idx += 1

        return chunks

    def _extract_sections(self, markdown: str) -> list[dict]:
        """Wyciągnij sekcje H2 z markdown."""
        sections = []
        current_header = ""
        current_lines = []

        for line in markdown.split("\n"):
            if line.startswith("## "):
                if current_header:
                    sections.append({
                        "header": current_header,
                        "content": "\n".join(current_lines).strip(),
                    })
                current_header = line.lstrip("# ").strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_header:
            sections.append({
                "header": current_header,
                "content": "\n".join(current_lines).strip(),
            })

        # Jeśli brak H2, cały tekst = jeden parent
        if not sections:
            sections = [{"header": "Artykuł", "content": markdown}]

        return sections
