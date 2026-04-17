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
        nazwa: Nazwa metody chunkingu.
        config: Opcjonalna konfiguracja chunkingu.
        embeddings: Obiekt LangChain Embeddings (wymagany dla 'semantic').

    Returns:
        Instancja ChunkerBase.
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
    elif nazwa == "proposition":
        from src.chunking.proposition import PropositionChunker
        return PropositionChunker()
    elif nazwa == "parent_child":
        from src.chunking.parent_child import ParentChildChunker
        return ParentChildChunker(cfg)
    elif nazwa == "sentence":
        from src.chunking.sentence import SentenceChunker
        return SentenceChunker()
    elif nazwa == "layout_aware":
        from src.chunking.layout_aware import LayoutAwareChunker
        return LayoutAwareChunker()
    else:
        raise ValueError(
            f"Nieznana metoda chunkingu: '{nazwa}'. "
            f"Dostępne: {AVAILABLE_METHODS}"
        )


# Lista dostępnych metod
AVAILABLE_METHODS = [
    "naive", "header", "semantic",
    "proposition", "parent_child", "sentence", "layout_aware",
]
