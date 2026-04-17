"""Proposition-based chunking — LLM rozbija tekst na atomowe twierdzenia.

Każde twierdzenie to osobny chunk: "VAT w Niemczech to 19%", "Próg OSS wynosi 10000 EUR".
Najwyższa granulacja — max precyzja retrieval.
"""

import json

import anthropic

from config.settings import get_claude_config
from src.chunking.base import Chunk, ChunkerBase

PROPOSITION_PROMPT = """Rozbij poniższy tekst na listę atomowych twierdzeń faktycznych.
Każde twierdzenie powinno być:
- Samowystarczalne (zrozumiałe bez kontekstu)
- Jedno zdanie
- Zawierać konkretny fakt, liczbę lub definicję

Tekst:
{text}

Zwróć JSON array:
["twierdzenie 1", "twierdzenie 2", ...]

Zwróć TYLKO JSON array."""


class PropositionChunker(ChunkerBase):
    """Podział tekstu na atomowe twierdzenia faktyczne (Claude)."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    @property
    def name(self) -> str:
        return "proposition"

    def chunk(self, markdown_text: str, metadata: dict | None = None) -> list[Chunk]:
        """Rozbij tekst na atomowe propozycje.

        Args:
            markdown_text: Tekst artykułu.
            metadata: Bazowe metadane.

        Returns:
            Lista Chunk — każdy = jedno atomowe twierdzenie.
        """
        base_meta = metadata or {}
        slug = base_meta.get("slug", "unknown")

        # Dziel tekst na fragmenty ~3000 znaków (limit kontekstu dla promptu)
        fragment_size = 3000
        fragments = []
        for i in range(0, len(markdown_text), fragment_size):
            fragments.append(markdown_text[i:i + fragment_size])

        all_propositions = []
        for fragment in fragments:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": PROPOSITION_PROMPT.format(text=fragment),
                }],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            try:
                props = json.loads(raw)
                if isinstance(props, list):
                    all_propositions.extend([str(p) for p in props if p])
            except json.JSONDecodeError:
                # Fallback: split po nowych liniach
                for line in raw.split("\n"):
                    line = line.strip().lstrip("- ").strip('"')
                    if line and len(line) > 10:
                        all_propositions.append(line)

        chunks = []
        for i, prop in enumerate(all_propositions):
            chunk_meta = {
                **base_meta,
                "method": self.name,
                "chunk_index": i,
                "char_count": len(prop),
            }
            chunks.append(Chunk(
                text=prop,
                metadata=chunk_meta,
                chunk_id=self._buduj_chunk_id(slug, i),
            ))

        return chunks

    def close(self):
        self._client.close()
