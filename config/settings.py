"""Centralna konfiguracja projektu RAG Pipeline.

Ładuje zmienne środowiskowe z .env i udostępnia je jako dataclassy.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Ładuj .env z katalogu głównego projektu
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


def _wymagana_zmienna(nazwa: str) -> str:
    """Pobierz zmienną środowiskową lub zgłoś jasny błąd."""
    wartosc = os.getenv(nazwa)
    if not wartosc:
        raise EnvironmentError(
            f"Brak zmiennej środowiskowej: {nazwa}. "
            f"Uzupełnij plik .env (wzór w .env.example)."
        )
    return wartosc


@dataclass
class QdrantConfig:
    """Konfiguracja Qdrant Cloud."""

    url: str = field(default_factory=lambda: _wymagana_zmienna("QDRANT_URL"))
    api_key: str = field(default_factory=lambda: _wymagana_zmienna("QDRANT_API_KEY"))
    collection_prefix: str = "articles"

    def collection_name(self, metoda: str) -> str:
        """Nazwa kolekcji dla danej metody chunkingu, np. 'articles_naive'."""
        return f"{self.collection_prefix}_{metoda}"


@dataclass
class JinaConfig:
    """Konfiguracja Jina AI (embeddingi + reader)."""

    api_key: str = field(default_factory=lambda: _wymagana_zmienna("JINA_API_KEY"))
    model: str = "jina-embeddings-v5-text-small"
    dimensions: int = 1024
    embed_url: str = "https://api.jina.ai/v1/embeddings"
    reader_url: str = "https://r.jina.ai/"
    max_batch_size: int = 100


@dataclass
class ClaudeConfig:
    """Konfiguracja Claude API (Anthropic)."""

    api_key: str = field(default_factory=lambda: _wymagana_zmienna("ANTHROPIC_API_KEY"))
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048


@dataclass
class ChunkingConfig:
    """Konfiguracja metod chunkingu."""

    # Naive
    naive_chunk_size: int = 1000
    naive_chunk_overlap: int = 200

    # Header-based
    header_split_on: list[tuple[str, str]] = field(
        default_factory=lambda: [("##", "h2"), ("###", "h3")]
    )
    header_max_chunk_size: int = 2000  # sub-split sekcji dłuższych

    # Semantic
    semantic_breakpoint_type: str = "percentile"
    semantic_breakpoint_threshold: int = 85


@dataclass
class ProjectPaths:
    """Ścieżki katalogów projektu."""

    root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw_articles(self) -> Path:
        return self.data / "raw"

    @property
    def results(self) -> Path:
        return self.data / "results"

    @property
    def article_urls_file(self) -> Path:
        return self.data / "article_urls.txt"

    @property
    def test_set_file(self) -> Path:
        return self.data / "test_set.json"


# --- Singleton-like dostęp do konfiguracji ---

def get_qdrant_config() -> QdrantConfig:
    return QdrantConfig()


def get_jina_config() -> JinaConfig:
    return JinaConfig()


def get_claude_config() -> ClaudeConfig:
    return ClaudeConfig()


def get_chunking_config() -> ChunkingConfig:
    return ChunkingConfig()


def get_paths() -> ProjectPaths:
    return ProjectPaths()
