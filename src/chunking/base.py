"""Bazowe klasy i modele danych dla modułu chunkingu.

Definiuje Chunk (fragment tekstu z metadanymi) i ChunkerBase (interfejs chunkerów).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Pojedynczy fragment tekstu z metadanymi.

    Attributes:
        text: Treść fragmentu.
        metadata: Dodatkowe informacje (źródło, nagłówek H2, metoda itp.).
        chunk_id: Unikalny identyfikator w formacie '{slug}_{metoda}_{indeks}'.
    """

    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return (
            f"Chunk(id='{self.chunk_id}', "
            f"len={len(self.text)}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )


class ChunkerBase(ABC):
    """Abstrakcyjna klasa bazowa dla chunkerów.

    Każda implementacja musi dostarczyć:
    - name: nazwa metody chunkingu
    - chunk(): podział tekstu na fragmenty
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nazwa metody chunkingu (np. 'naive', 'header', 'semantic')."""
        ...

    @abstractmethod
    def chunk(self, markdown_text: str, metadata: dict | None = None) -> list[Chunk]:
        """Podziel tekst markdown na fragmenty.

        Args:
            markdown_text: Tekst artykułu w formacie markdown.
            metadata: Bazowe metadane do dołączenia do każdego chunka
                      (np. source_file, slug).

        Returns:
            Lista obiektów Chunk.
        """
        ...

    def _buduj_chunk_id(self, slug: str, indeks: int) -> str:
        """Generuje unikalny ID chunka."""
        return f"{slug}_{self.name}_{indeks}"
