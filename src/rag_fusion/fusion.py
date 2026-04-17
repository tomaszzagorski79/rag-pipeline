"""RAG-Fusion — multi-query retrieval z Reciprocal Rank Fusion.

Generuje N wariantów pytania z różnych perspektyw,
odpala retrieval dla każdego osobno, łączy wyniki przez RRF.
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config
from src.retrieval.hybrid_search import HybridRetriever, SearchResult

GENERATE_QUERIES_PROMPT = """Wygeneruj {n} różnych wariantów poniższego pytania.
Każdy wariant powinien ujmować pytanie z INNEJ perspektywy lub używać INNYCH słów kluczowych.
Zachowaj intencję, ale zmień formę.

Oryginalne pytanie: {query}

Zwróć JSON array:
["wariant 1", "wariant 2", ...]

Zwróć TYLKO JSON, bez komentarzy."""


@dataclass
class FusionResult:
    """Wynik RAG-Fusion."""

    original_query: str
    sub_queries: list[str] = field(default_factory=list)
    per_query_results: dict[str, list[SearchResult]] = field(default_factory=dict)
    fused_results: list[SearchResult] = field(default_factory=list)
    answer: str = ""


class RAGFusion:
    """RAG-Fusion — multi-query z RRF fusion."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        n_queries: int = 4,
        rrf_k: int = 60,
    ):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model
        self._retriever = retriever or HybridRetriever()
        self._n_queries = n_queries
        self._rrf_k = rrf_k

    def _generate_sub_queries(self, query: str) -> list[str]:
        """Wygeneruj N wariantów pytania."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            messages=[
                {"role": "user", "content": GENERATE_QUERIES_PROMPT.format(query=query, n=self._n_queries)},
            ],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            queries = json.loads(raw)
            if isinstance(queries, list):
                return [str(q) for q in queries]
        except json.JSONDecodeError:
            pass
        return [query]

    def _rrf_fusion(
        self,
        per_query_results: dict[str, list[SearchResult]],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion na wynikach z wielu zapytań.

        Score(d) = Σ 1/(k + rank_i(d)) dla każdej listy i.
        """
        # Zbierz RRF score per chunk_id
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, SearchResult] = {}

        for query_name, results in per_query_results.items():
            for rank, result in enumerate(results):
                cid = result.chunk_id or result.text[:50]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self._rrf_k + rank + 1)
                if cid not in chunk_data:
                    chunk_data[cid] = result

        # Sortuj po RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused = []
        for cid in sorted_ids[:top_k]:
            original = chunk_data[cid]
            fused.append(SearchResult(
                text=original.text,
                score=rrf_scores[cid],
                metadata={
                    **original.metadata,
                    "rrf_score": rrf_scores[cid],
                    "appeared_in_n_queries": sum(
                        1 for results in per_query_results.values()
                        if any(r.chunk_id == cid or r.text[:50] == cid for r in results)
                    ),
                },
                chunk_id=original.chunk_id,
            ))

        return fused

    def run(
        self,
        query: str,
        collection_name: str,
        top_k_per_query: int = 10,
        top_k_final: int = 5,
    ) -> FusionResult:
        """Uruchom RAG-Fusion pipeline.

        Args:
            query: Pytanie użytkownika.
            collection_name: Nazwa kolekcji Qdrant.
            top_k_per_query: Ile wyników per sub-query.
            top_k_final: Ile wyników po fusion.

        Returns:
            FusionResult z sub-queries, wynikami per query i fused results.
        """
        result = FusionResult(original_query=query)

        # 1. Generuj sub-queries
        sub_queries = self._generate_sub_queries(query)
        # Dodaj oryginalne pytanie
        all_queries = [query] + sub_queries
        result.sub_queries = all_queries

        # 2. Retrieval per query
        for q in all_queries:
            results = self._retriever.search(q, collection_name, limit=top_k_per_query)
            result.per_query_results[q] = results

        # 3. RRF Fusion
        result.fused_results = self._rrf_fusion(result.per_query_results, top_k=top_k_final)

        return result

    def close(self):
        self._client.close()
        self._retriever.store.close()
