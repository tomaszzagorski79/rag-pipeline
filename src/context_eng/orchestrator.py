"""Orchestrator — Claude decyduje które źródła retrieval użyć.

Komponent 1 Context Engineering: agent/orkiestrator w runtime.
Zamiast checkboxów użytkownika, Claude analizuje pytanie i dostępne źródła,
i wybiera optymalną kombinację.
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config

ORCHESTRATOR_PROMPT = """Jesteś orkiestratorem systemu retrieval. Masz dostęp do {num_sources} źródeł:

{sources_desc}

Pytanie użytkownika: {query}
Typ pytania: {intent}

Twoje zadanie: zdecyduj które źródła wywołać, biorąc pod uwagę:
- Faktyczne pytania → vector + graph (jeśli dostępne)
- Porównania/analizy → vector + graph + pageindex (jeśli wszystkie dostępne)
- Eksploracyjne → wszystkie dostępne
- Proste lookup → tylko vector
- Pytania o konkretny dokument → pageindex (jeśli dostępny)

Zwróć JSON:
{{
  "use_vector": true/false,
  "use_graph": true/false,
  "use_pageindex": true/false,
  "reasoning": "krótkie wyjaśnienie wyboru"
}}

Zwróć TYLKO JSON, bez komentarzy."""


SOURCE_DESCRIPTIONS = {
    "vector": "VECTOR (Qdrant hybrid search) — semantyczne podobieństwo, dobry dla większości pytań",
    "graph": "GRAPH (Neo4j EAV) — relacje między encjami, dobry dla pytań faktograficznych i porównawczych",
    "pageindex": "PAGEINDEX — reasoning-based, dobry dla pytań wymagających nawigacji po strukturze dokumentu",
}


@dataclass
class OrchestratorDecision:
    """Decyzja orkiestratora o użyciu źródeł."""

    use_vector: bool = False
    use_graph: bool = False
    use_pageindex: bool = False
    reasoning: str = ""
    sources_available: dict[str, bool] = field(default_factory=dict)


class Orchestrator:
    """Orkiestrator — Claude decyduje które źródła użyć."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def decide(
        self,
        query: str,
        available_sources: dict[str, bool],
        intent: str = "FACTUAL",
    ) -> OrchestratorDecision:
        """Zdecyduj które źródła wywołać.

        Args:
            query: Pytanie użytkownika.
            available_sources: dict {"vector": True/False, "graph": ..., "pageindex": ...}
            intent: Typ pytania z QueryAugmenter.

        Returns:
            OrchestratorDecision z flagami użyj/nie używaj.
        """
        # Opis tylko dostępnych źródeł
        available = [k for k, v in available_sources.items() if v]
        if not available:
            return OrchestratorDecision(
                reasoning="Brak dostępnych źródeł.",
                sources_available=available_sources,
            )

        sources_desc = "\n".join(
            f"- {SOURCE_DESCRIPTIONS.get(src, src)}"
            for src in available
        )

        prompt = ORCHESTRATOR_PROMPT.format(
            num_sources=len(available),
            sources_desc=sources_desc,
            query=query,
            intent=intent,
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            data = json.loads(raw)
            decision = OrchestratorDecision(
                use_vector=bool(data.get("use_vector", False)) and available_sources.get("vector", False),
                use_graph=bool(data.get("use_graph", False)) and available_sources.get("graph", False),
                use_pageindex=bool(data.get("use_pageindex", False)) and available_sources.get("pageindex", False),
                reasoning=str(data.get("reasoning", "")),
                sources_available=available_sources,
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback: wszystkie dostępne
            decision = OrchestratorDecision(
                use_vector=available_sources.get("vector", False),
                use_graph=available_sources.get("graph", False),
                use_pageindex=available_sources.get("pageindex", False),
                reasoning="Fallback: użyj wszystkich dostępnych źródeł (błąd parsowania orkiestratora).",
                sources_available=available_sources,
            )

        return decision

    def close(self):
        self._client.close()
