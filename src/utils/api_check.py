"""Sprawdzanie statusu kluczy API i dostępności zewnętrznych usług."""

import os
from dataclasses import dataclass


@dataclass
class APIStatus:
    """Status jednej usługi API."""

    name: str
    env_key: str
    is_set: bool
    is_required: bool  # True = pipeline nie zadziała bez tego
    description: str
    setup_url: str
    enables: list[str]  # zakładki które wymagają tego klucza


def check_all_apis() -> list[APIStatus]:
    """Sprawdź status wszystkich kluczy API.

    Returns:
        Lista APIStatus z informacją co jest ustawione, co wymagane.
    """
    return [
        APIStatus(
            name="Qdrant Cloud",
            env_key="QDRANT_API_KEY",
            is_set=bool(os.getenv("QDRANT_API_KEY")) and bool(os.getenv("QDRANT_URL")),
            is_required=True,
            description="Vector DB — przechowywanie i wyszukiwanie chunków. Bez tego 14/22 zakładek nie działa.",
            setup_url="https://cloud.qdrant.io/",
            enables=[
                "2. Embeddingi tytułów", "3. Chunking", "4. Zapytania",
                "5. Re-ranking", "6. HyDE", "7. Adaptive RAG", "8. CRAG",
                "9. Ewaluacja RAGAS", "10. Hallucination (generuj)",
                "14. Agentic RAG", "16. Hybrid V+G", "17. Context Engineering",
                "18. RAG-Fusion", "19. FLARE", "21. Speculative RAG",
            ],
        ),
        APIStatus(
            name="Jina AI",
            env_key="JINA_API_KEY",
            is_set=bool(os.getenv("JINA_API_KEY")),
            is_required=True,
            description="Jina Reader (scraping) + Jina Embeddings v5 (dense vectors 1024d).",
            setup_url="https://jina.ai/",
            enables=[
                "1. Scraping", "2. Embeddingi tytułów", "3. Chunking (dense vectors)",
                "4+ wszystkie RAG (retrieval)", "11. Benchmarki (baseline Jina)",
            ],
        ),
        APIStatus(
            name="Anthropic (Claude)",
            env_key="ANTHROPIC_API_KEY",
            is_set=bool(os.getenv("ANTHROPIC_API_KEY")),
            is_required=True,
            description="Claude Sonnet 4 — LLM do generowania odpowiedzi. Claude Haiku 4 — drafter w Speculative RAG.",
            setup_url="https://console.anthropic.com/",
            enables=[
                "4+ wszystkie zakładki RAG (generowanie odpowiedzi)",
                "7. Classification, 8. Reformulation, 10. Verification",
                "13-15, 17, 20, 21 (chunkowanie/analizy)",
            ],
        ),
        APIStatus(
            name="Google AI (Gemini)",
            env_key="GOOGLE_API_KEY",
            is_set=bool(os.getenv("GOOGLE_API_KEY")),
            is_required=False,
            description="OPCJONALNE — tylko do benchmarków embeddingów Gemini.",
            setup_url="https://aistudio.google.com/apikey",
            enables=["11. Benchmarki embeddingów (model Gemini)"],
        ),
        APIStatus(
            name="Neo4j Aura",
            env_key="NEO4J_PASSWORD",
            is_set=bool(os.getenv("NEO4J_PASSWORD")) and bool(os.getenv("NEO4J_URI")),
            is_required=False,
            description="OPCJONALNE — graph database dla Graph RAG, Hybrid V+G, Context Engineering (graph source).",
            setup_url="https://console.neo4j.io/",
            enables=["15. Graph RAG", "16. Hybrid Vector+Graph", "17. Context Eng (graph)"],
        ),
    ]


def get_missing_required() -> list[APIStatus]:
    """Zwróć listę WYMAGANYCH kluczy których brakuje."""
    return [s for s in check_all_apis() if s.is_required and not s.is_set]


def is_qdrant_available() -> bool:
    """Sprawdź czy Qdrant jest skonfigurowany."""
    return bool(os.getenv("QDRANT_URL")) and bool(os.getenv("QDRANT_API_KEY"))


def is_anthropic_available() -> bool:
    """Sprawdź czy Anthropic API jest skonfigurowane."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def is_jina_available() -> bool:
    """Sprawdź czy Jina API jest skonfigurowane."""
    return bool(os.getenv("JINA_API_KEY"))


def is_neo4j_available() -> bool:
    """Sprawdź czy Neo4j jest skonfigurowany."""
    return bool(os.getenv("NEO4J_URI")) and bool(os.getenv("NEO4J_PASSWORD"))
