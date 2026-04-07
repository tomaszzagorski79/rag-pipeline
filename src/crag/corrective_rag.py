"""CRAG — Corrective Retrieval-Augmented Generation.

Po retrieval sprawdza jakość wyników. Jeśli słabe — przeformułowuje pytanie
i szuka ponownie. Jeśli nadal słabe — odmawia odpowiedzi.
"""

from dataclasses import dataclass, field

from src.generation.claude_gen import ClaudeGenerator
from src.retrieval.hybrid_search import HybridRetriever, SearchResult

REFORMULATE_PROMPT = """Poniższe pytanie nie zwróciło dobrych wyników w wyszukiwaniu.
Przeformułuj je — użyj innych słów kluczowych, synonimów, bardziej precyzyjnego języka.
Zwróć TYLKO przeformułowane pytanie, bez wyjaśnień.

Oryginalne pytanie: {query}
"""


@dataclass
class CRAGStep:
    """Pojedynczy krok w łańcuchu CRAG."""

    step_name: str
    query_used: str
    best_score: float
    num_results: int
    decision: str  # "accept", "reformulate", "reject"


@dataclass
class CRAGResult:
    """Wynik pełnego pipeline'u CRAG."""

    steps: list[CRAGStep] = field(default_factory=list)
    final_answer: str = ""
    final_contexts: list[str] = field(default_factory=list)
    final_results: list[SearchResult] = field(default_factory=list)
    was_reformulated: bool = False
    was_rejected: bool = False


class CorrectiveRAG:
    """Pipeline CRAG — retrieval z korekcją jakości wyników."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: ClaudeGenerator | None = None,
        score_threshold: float = 0.02,
        max_retries: int = 1,
    ):
        """Inicjalizacja CRAG.

        Args:
            retriever: Retriever do wyszukiwania.
            generator: Generator odpowiedzi (Claude).
            score_threshold: Minimalny score żeby zaakceptować wyniki.
                             UWAGA: RRF score to ~0.01-0.08, NIE 0-1.
            max_retries: Ile razy przeformułować pytanie (domyślnie 1).
        """
        self._retriever = retriever or HybridRetriever()
        self._generator = generator or ClaudeGenerator()
        self._threshold = score_threshold
        self._max_retries = max_retries

    def _reformulate_query(self, query: str) -> str:
        """Przeformułuj pytanie przez Claude."""
        import anthropic
        from config.settings import get_claude_config

        cfg = get_claude_config()
        client = anthropic.Anthropic(api_key=cfg.api_key)
        response = client.messages.create(
            model=cfg.model,
            max_tokens=200,
            messages=[
                {"role": "user", "content": REFORMULATE_PROMPT.format(query=query)},
            ],
        )
        client.close()
        return response.content[0].text.strip()

    def run(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
    ) -> CRAGResult:
        """Uruchom pipeline CRAG.

        Args:
            query: Pytanie użytkownika.
            collection_name: Nazwa kolekcji Qdrant.
            limit: Liczba kontekstów.

        Returns:
            CRAGResult z łańcuchem decyzji i finalną odpowiedzią.
        """
        result = CRAGResult()
        current_query = query

        for attempt in range(1 + self._max_retries):
            # Retrieval
            search_results = self._retriever.search(
                current_query, collection_name, limit=limit
            )

            best_score = search_results[0].score if search_results else 0.0

            # Decyzja
            if best_score >= self._threshold and search_results:
                decision = "accept"
            elif attempt < self._max_retries:
                decision = "reformulate"
            else:
                decision = "reject"

            step = CRAGStep(
                step_name=f"Próba {attempt + 1}",
                query_used=current_query,
                best_score=best_score,
                num_results=len(search_results),
                decision=decision,
            )
            result.steps.append(step)

            if decision == "accept":
                # Generuj odpowiedź
                contexts = [r.text for r in search_results]
                result.final_answer = self._generator.generate(query, contexts)
                result.final_contexts = contexts
                result.final_results = search_results
                return result

            elif decision == "reformulate":
                result.was_reformulated = True
                current_query = self._reformulate_query(current_query)

            elif decision == "reject":
                result.was_rejected = True
                result.final_answer = (
                    "Na podstawie dostępnych fragmentów nie mogę udzielić "
                    "wiarygodnej odpowiedzi na to pytanie. Wyniki wyszukiwania "
                    "były zbyt słabo dopasowane do zapytania."
                )
                result.final_results = search_results
                return result

        return result
