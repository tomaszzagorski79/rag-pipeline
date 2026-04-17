"""Layout-aware chunking — rozpoznaje tabele, listy i nagłówki w markdown.

Nie tnie w środku tabeli ani listy. Zachowuje strukturalne elementy w całości.
"""

import re

from src.chunking.base import Chunk, ChunkerBase


class LayoutAwareChunker(ChunkerBase):
    """Podział tekstu z zachowaniem struktury markdown (tabele, listy, bloki kodu)."""

    def __init__(self, max_chunk_size: int = 2000):
        self._max_size = max_chunk_size

    @property
    def name(self) -> str:
        return "layout_aware"

    def chunk(self, markdown_text: str, metadata: dict | None = None) -> list[Chunk]:
        """Podziel tekst zachowując strukturę (tabele, listy, nagłówki).

        Args:
            markdown_text: Tekst artykułu w markdown.
            metadata: Bazowe metadane.

        Returns:
            Lista Chunk z zachowaną strukturą.
        """
        base_meta = metadata or {}
        slug = base_meta.get("slug", "unknown")

        # Identyfikuj bloki strukturalne
        blocks = self._extract_blocks(markdown_text)

        # Łącz małe bloki, nie łam dużych
        chunks = []
        current_text = ""
        current_header = ""
        idx = 0

        for block in blocks:
            # Jeśli blok jest za duży sam w sobie — daj jako osobny chunk
            if len(block["text"]) > self._max_size:
                # Flush current
                if current_text.strip():
                    chunks.append(self._make_chunk(
                        current_text.strip(), base_meta, slug, idx, current_header
                    ))
                    idx += 1
                    current_text = ""

                chunks.append(self._make_chunk(
                    block["text"], base_meta, slug, idx, block.get("header", current_header)
                ))
                idx += 1
                continue

            # Sprawdź czy dodanie bloku przekroczy limit
            if len(current_text) + len(block["text"]) > self._max_size and current_text.strip():
                chunks.append(self._make_chunk(
                    current_text.strip(), base_meta, slug, idx, current_header
                ))
                idx += 1
                current_text = ""

            # Aktualizuj nagłówek
            if block.get("header"):
                current_header = block["header"]

            current_text += "\n\n" + block["text"]

        # Flush ostatni
        if current_text.strip():
            chunks.append(self._make_chunk(
                current_text.strip(), base_meta, slug, idx, current_header
            ))

        return chunks

    def _make_chunk(self, text: str, base_meta: dict, slug: str, idx: int, header: str) -> Chunk:
        """Utwórz Chunk z metadanymi."""
        meta = {
            **base_meta,
            "method": self.name,
            "chunk_index": idx,
            "char_count": len(text),
        }
        if header:
            meta["h2"] = header
        return Chunk(text=text, metadata=meta, chunk_id=self._buduj_chunk_id(slug, idx))

    def _extract_blocks(self, markdown: str) -> list[dict]:
        """Wyciągnij bloki strukturalne z markdown."""
        blocks = []
        lines = markdown.split("\n")

        current_block_lines = []
        current_type = "text"
        current_header = ""
        in_table = False
        in_list = False

        for line in lines:
            stripped = line.strip()

            # Nagłówek H2/H3 — nowy blok
            if re.match(r"^#{2,3}\s+", line):
                if current_block_lines:
                    blocks.append({
                        "text": "\n".join(current_block_lines).strip(),
                        "type": current_type,
                        "header": current_header,
                    })
                current_block_lines = [line]
                current_header = stripped.lstrip("# ").strip()
                current_type = "header"
                in_table = False
                in_list = False
                continue

            # Tabela (linia zaczyna się od |)
            if stripped.startswith("|"):
                if not in_table and current_block_lines and current_type != "table":
                    blocks.append({
                        "text": "\n".join(current_block_lines).strip(),
                        "type": current_type,
                        "header": current_header,
                    })
                    current_block_lines = []
                in_table = True
                in_list = False
                current_type = "table"
                current_block_lines.append(line)
                continue

            # Koniec tabeli
            if in_table and not stripped.startswith("|"):
                blocks.append({
                    "text": "\n".join(current_block_lines).strip(),
                    "type": "table",
                    "header": current_header,
                })
                current_block_lines = []
                in_table = False

            # Lista (*, -, 1.)
            if re.match(r"^[\*\-]\s|^\d+\.\s", stripped):
                if not in_list and current_block_lines and current_type != "list":
                    blocks.append({
                        "text": "\n".join(current_block_lines).strip(),
                        "type": current_type,
                        "header": current_header,
                    })
                    current_block_lines = []
                in_list = True
                current_type = "list"
                current_block_lines.append(line)
                continue

            # Koniec listy (pusta linia po liście)
            if in_list and stripped == "":
                blocks.append({
                    "text": "\n".join(current_block_lines).strip(),
                    "type": "list",
                    "header": current_header,
                })
                current_block_lines = []
                in_list = False
                current_type = "text"
                continue

            # Zwykły tekst
            if not in_table and not in_list:
                current_type = "text"
            current_block_lines.append(line)

        # Flush
        if current_block_lines:
            blocks.append({
                "text": "\n".join(current_block_lines).strip(),
                "type": current_type,
                "header": current_header,
            })

        return [b for b in blocks if b["text"]]
