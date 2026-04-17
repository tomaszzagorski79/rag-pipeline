"""Adaptive RAG — dynamiczny routing pytań przez różne strategie retrieval.

Proste pytania → lekki pipeline (top-3, direct).
Średnie → standardowy RAG (top-5).
Złożone → pełny pipeline (top-10 + HyDE + re-ranking).
"""

from dataclasses import dataclass, field

from src.adaptive.query_classifier import QueryClassifier
from src.generation.claude_gen import ClaudeGenerator
from src.hyde.hyde_generator import HyDEGenerator
from src.reranking.flashrank_reranker import FlashRankReranker
from src.retrieval.hybrid_search import HybridRetriever, SearchResult


@dataclass
class AdaptiveResult:
    """Wynik Adaptive RAG z informacją o wybranej strategii."""

    classification: str  # simple/medium/complex
    strategy_description: str
    contexts: list[str] = field(default_factory=list)
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""
    hyde_hypothesis: str | None = None
    was_reranked: bool = False


STRATEGIES = {
    "no_retrieval": {
        "top_k": 0,
        "use_hyde": False,
        "use_reranking": False,
        "description": "NO RETRIEVAL: LLM odpowiada z pamięci (wiedza ogólna, definicje). Zero kosztów retrieval.",
    },
    "simple": {
        "top_k": 3,
        "use_hyde": False,
        "use_reranking": False,
        "description": "SIMPLE: top-3 kontekstów, bezpośrednia odpowiedź. Szybko i tanio.",
    },
    "medium": {
        "top_k": 5,
        "use_hyde": False,
        "use_reranking": False,
        "description": "MEDIUM: top-5 kontekstów, standardowy RAG.",
    },
    "complex": {
        "top_k": 10,
        "use_hyde": True,
        "use_reranking": True,
        "description": "COMPLEX: top-10 + HyDE + re-ranking. Pełny pipeline dla złożonych pytań.",
    },
}


class AdaptiveRAGPipeline:
    """Pipeline Adaptive RAG z dynamicznym routingiem."""

    def __init__(
        self,
        classifier: QueryClassifier | None = None,
        retriever: HybridRetriever | None = None,
        generator: ClaudeGenerator | None = None,
        reranker: FlashRankReranker | None = None,
        hyde: HyDEGenerator | None = None,
    ):
        self._classifier = classifier or QueryClassifier()
        self._retriever = retriever or HybridRetriever()
        self._generator = generator or ClaudeGenerator()
        self._reranker = reranker or FlashRankReranker()
        self._hyde = hyde or HyDEGenerator()

    def run(
        self,
        query: str,
        collection_name: str,
        override_classification: str | None = None,
    ) -> AdaptiveResult:
        """Uruchom Adaptive RAG.

        Args:
            query: Pytanie użytkownika.
            collection_name: Nazwa kolekcji Qdrant.
            override_classification: Opcjonalne wymuszenie klasyfikacji.

        Returns:
            AdaptiveResult z pełnym wynikiem.
        """
        # 1. Klasyfikacja
        classification = override_classification or self._classifier.classify(query)
        strategy = STRATEGIES.get(classification, STRATEGIES["medium"])

        result = AdaptiveResult(
            classification=classification,
            strategy_description=strategy["description"],
        )

        top_k = strategy["top_k"]

        # 2a. No retrieval — LLM odpowiada z pamięci
        if top_k == 0:
            import anthropic
            from config.settings import get_claude_config
            cfg = get_claude_config()
            client = anthropic.Anthropic(api_key=cfg.api_key)
            response = client.messages.create(
                model=cfg.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": query}],
            )
            client.close()
            result.answer = response.content[0].text
            return result

        # 2. Retrieval (standardowy lub HyDE)
        if strategy["use_hyde"]:
            hypothesis, hyde_results = self._hyde.search_with_hyde(
                query, collection_name, limit=top_k
            )
            result.hyde_hypothesis = hypothesis
            # Łącz wyniki HyDE z hybrid search
            hybrid_results = self._retriever.search(query, collection_name, limit=top_k)
            # Dedup po chunk_id, zachowaj lepszy score
            seen = {}
            for r in hyde_results + hybrid_results:
                if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                    seen[r.chunk_id] = r
            all_results = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]
        else:
            all_results = self._retriever.search(query, collection_name, limit=top_k)

        # 3. Re-ranking (opcjonalnie)
        if strategy["use_reranking"] and all_results:
            all_results = self._reranker.rerank(query, all_results, top_k=min(5, top_k))
            result.was_reranked = True

        result.results = all_results
        result.contexts = [r.text for r in all_results]

        # 4. Generation
        result.answer = self._generator.generate(query, result.contexts)

        return result
