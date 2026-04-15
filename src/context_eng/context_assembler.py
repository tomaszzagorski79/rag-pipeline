"""Context Engineering — zarządzanie pełnym kontekstem LLM.

Framework: agent + wieloźródłowy retrieval + context budget + faceted search.

Komponenty:
1. Agent/orkiestrator — decyduje skąd pobrać
2. Wieloźródłowy retrieval (Qdrant + Neo4j)
3. Faceted search (metadane o krajobrazie danych)
4. Context window budgeting (dzielenie tokenów)
5. Memory wielowarstwowa (short-term + episodic)
"""

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class FacetedMetadata:
    """Metadane "peryferyjnego widzenia" — co jeszcze jest dostępne."""

    total_found: int = 0
    by_source: dict[str, int] = field(default_factory=dict)  # vector/graph/pageindex
    by_article: dict[str, int] = field(default_factory=dict)  # slug -> count
    by_section: dict[str, int] = field(default_factory=dict)  # h2 -> count


@dataclass
class ContextBudget:
    """Budżet tokenów w kontekście LLM."""

    total: int = 100000
    system_prompt: int = 500
    history: int = 5000
    retrieval: int = 80000
    response_reserve: int = 4000

    def available_for_retrieval(self) -> int:
        return self.total - self.system_prompt - self.history - self.response_reserve


@dataclass
class AssembledContext:
    """Zmontowany kontekst do wysłania do LLM."""

    query: str
    facets: FacetedMetadata = field(default_factory=FacetedMetadata)
    ranked_contexts: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    total_tokens_estimate: int = 0
    budget: ContextBudget = field(default_factory=ContextBudget)


def _estimate_tokens(text: str) -> int:
    """Szybkie oszacowanie tokenów (~4 znaki na token dla polskiego)."""
    return len(text) // 4


class ContextAssembler:
    """Assembler kontekstu — łączy retrieval z wielu źródeł w jeden LLM context."""

    def __init__(self, budget: ContextBudget | None = None):
        self._budget = budget or ContextBudget()

    def assemble(
        self,
        query: str,
        vector_results: list = None,
        graph_contexts: list = None,
        pageindex_nodes: list = None,
    ) -> AssembledContext:
        """Złóż kontekst z wielu źródeł z budżetowaniem tokenów.

        Args:
            query: Pytanie użytkownika.
            vector_results: Lista SearchResult z Qdrant.
            graph_contexts: Lista GraphContext z Neo4j.
            pageindex_nodes: Lista TreeNode z PageIndex.

        Returns:
            AssembledContext z priorytetyzowanym, budżetowanym kontekstem.
        """
        assembled = AssembledContext(query=query, budget=self._budget)

        # Zbierz wszystkie fragmenty z metadanymi
        all_fragments = []  # (score, text, source, slug, section)

        if vector_results:
            for r in vector_results:
                all_fragments.append({
                    "score": r.score,
                    "text": r.text,
                    "source": "vector",
                    "slug": r.metadata.get("slug", "?"),
                    "section": r.metadata.get("h2", ""),
                })
            assembled.facets.by_source["vector"] = len(vector_results)
            assembled.sources_used.append("vector")

        if graph_contexts:
            graph_count = 0
            for ctx in graph_contexts:
                for m in ctx.mentions:
                    attrs_str = "; ".join(
                        f"{a.get('attribute')}={a.get('value')}"
                        for a in m.get("attrs", []) if a.get("attribute")
                    )
                    text_with_attrs = m.get("text", "")
                    if attrs_str:
                        text_with_attrs = f"[Entity {ctx.entity} — {attrs_str}] {text_with_attrs}"
                    all_fragments.append({
                        "score": 0.5,  # placeholder — graf nie ma score
                        "text": text_with_attrs,
                        "source": "graph",
                        "slug": m.get("slug", "?"),
                        "section": "",
                    })
                    graph_count += 1
            assembled.facets.by_source["graph"] = graph_count
            if graph_count > 0:
                assembled.sources_used.append("graph")

        if pageindex_nodes:
            for node in pageindex_nodes:
                all_fragments.append({
                    "score": 0.7,
                    "text": node.content,
                    "source": "pageindex",
                    "slug": "pageindex",
                    "section": node.title,
                })
            assembled.facets.by_source["pageindex"] = len(pageindex_nodes)
            if pageindex_nodes:
                assembled.sources_used.append("pageindex")

        # Facety: by_article i by_section
        articles = Counter(f["slug"] for f in all_fragments)
        sections = Counter(f["section"] for f in all_fragments if f["section"])
        assembled.facets.by_article = dict(articles)
        assembled.facets.by_section = dict(sections)
        assembled.facets.total_found = len(all_fragments)

        # Sortuj po score i dedupuj
        all_fragments.sort(key=lambda x: x["score"], reverse=True)

        # Budżetowanie tokenów
        budget_available = self._budget.available_for_retrieval()
        used_tokens = 0
        seen_texts = set()

        for frag in all_fragments:
            text = frag["text"]
            text_hash = text[:100]  # prosty dedup
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)

            tokens = _estimate_tokens(text)
            if used_tokens + tokens > budget_available:
                break

            # Formatuj z metadanymi
            formatted = f"[{frag['source'].upper()}]"
            if frag["slug"] != "?":
                formatted += f" ({frag['slug']})"
            formatted += f"\n{text}"

            assembled.ranked_contexts.append(formatted)
            used_tokens += tokens + _estimate_tokens(formatted) - tokens

        assembled.total_tokens_estimate = used_tokens

        return assembled
