"""Rejestr chunkerów — factory pattern.

Umożliwia pobranie chunkera po nazwie lub wszystkich naraz.
"""

from config.settings import ChunkingConfig, get_chunking_config
from src.chunking.base import ChunkerBase
from src.chunking.header_based import HeaderChunker
from src.chunking.naive import NaiveChunker
from src.chunking.semantic import SemanticChunkerWrapper


def get_chunker(
    nazwa: str,
    config: ChunkingConfig | None = None,
    embeddings=None,
) -> ChunkerBase:
    """Zwraca chunker po nazwie.

    Args:
        nazwa: 'naive', 'header' lub 'semantic'.
        config: Opcjonalna konfiguracja chunkingu.
        embeddings: Obiekt LangChain Embeddings (wymagany dla 'semantic').

    Returns:
        Instancja ChunkerBase.

    Raises:
        ValueError: Gdy nazwa jest nieznana.
        ValueError: Gdy brak embeddings dla 'semantic'.
    """
    cfg = config or get_chunking_config()

    if nazwa == "naive":
        return NaiveChunker(cfg)
    elif nazwa == "header":
        return HeaderChunker(cfg)
    elif nazwa == "semantic":
        if embeddings is None:
            raise ValueError(
                "SemanticChunker wymaga obiektu embeddings "
                "(np. JinaEmbeddingsLangChain)."
            )
        return SemanticChunkerWrapper(embeddings=embeddings, config=cfg)
    else:
        raise ValueError(
            f"Nieznana metoda chunkingu: '{nazwa}'. "
            f"Dostępne: naive, header, semantic."
        )


def get_all_chunkers(
    config: ChunkingConfig | None = None,
    embeddings=None,
) -> dict[str, ChunkerBase]:
    """Zwraca słownik wszystkich dostępnych chunkerów.

    Args:
        config: Opcjonalna konfiguracja chunkingu.
        embeddings: Obiekt LangChain Embeddings (wymagany dla 'semantic').

    Returns:
        dict[str, ChunkerBase] np. {'naive': NaiveChunker(...), ...}
    """
    chunkers = {
        "naive": get_chunker("naive", config),
        "header": get_chunker("header", config),
    }

    if embeddings is not None:
        chunkers["semantic"] = get_chunker("semantic", config, embeddings)

    return chunkers


# Lista dostępnych metod
AVAILABLE_METHODS = ["naive", "header", "semantic"]
