"""Speculative RAG — drafter generuje N draftów równolegle, verifier wybiera najlepszy.

Drafter: Claude Haiku (szybki, tani) — generuje kandydackie odpowiedzi z różnych podzbiorów kontekstu.
Verifier: Claude Sonnet (dokładny) — ocenia i wybiera najlepszą.
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config

DRAFT_PROMPT = """Odpowiedz na pytanie TYLKO na podstawie poniższego kontekstu.
Odpowiadaj po polsku, zwięźle (3-5 zdań). Cytuj źródła.

Kontekst:
{context}

Pytanie: {query}

Odpowiedź:"""

VERIFY_PROMPT = """Oceń poniższe drafty odpowiedzi na pytanie. Dla każdego oceń:
- faithfulness (czy oparty na kontekście): 1-10
- relevance (czy odpowiada na pytanie): 1-10
- completeness (czy kompletny): 1-10

Pytanie: {query}

{drafts}

Zwróć JSON:
[
  {{"draft_id": 1, "faithfulness": 8, "relevance": 9, "completeness": 7, "total": 24}},
  ...
]

Zwróć TYLKO JSON array."""


@dataclass
class Draft:
    """Pojedynczy draft odpowiedzi."""

    draft_id: int
    text: str
    context_subset: list[str] = field(default_factory=list)
    faithfulness: int = 0
    relevance: int = 0
    completeness: int = 0
    total_score: int = 0
    is_selected: bool = False


@dataclass
class SpeculativeResult:
    """Wynik Speculative RAG."""

    query: str
    drafts: list[Draft] = field(default_factory=list)
    selected_draft: Draft | None = None
    final_answer: str = ""


class SpeculativeRAG:
    """Speculative RAG — drafter + verifier pattern."""

    def __init__(self, n_drafts: int = 3):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._drafter_model = "claude-haiku-4-20250414"
        self._verifier_model = cfg.model  # Sonnet
        self._n_drafts = n_drafts

    def _generate_draft(self, query: str, context_subset: list[str]) -> str:
        """Generuj draft z podzbioru kontekstu (Haiku — szybki)."""
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_subset))
        response = self._client.messages.create(
            model=self._drafter_model,
            max_tokens=500,
            messages=[{"role": "user", "content": DRAFT_PROMPT.format(context=ctx[:6000], query=query)}],
        )
        return response.content[0].text.strip()

    def _verify_drafts(self, query: str, drafts: list[Draft]) -> list[Draft]:
        """Verifier ocenia wszystkie drafty (Sonnet — dokładny)."""
        drafts_text = "\n\n".join(
            f"--- DRAFT {d.draft_id} ---\n{d.text}" for d in drafts
        )

        response = self._client.messages.create(
            model=self._verifier_model,
            max_tokens=500,
            messages=[{"role": "user", "content": VERIFY_PROMPT.format(query=query, drafts=drafts_text)}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            scores = json.loads(raw)
            for score_data in scores:
                did = score_data.get("draft_id", 0)
                for d in drafts:
                    if d.draft_id == did:
                        d.faithfulness = score_data.get("faithfulness", 5)
                        d.relevance = score_data.get("relevance", 5)
                        d.completeness = score_data.get("completeness", 5)
                        d.total_score = score_data.get("total", 15)
                        break
        except json.JSONDecodeError:
            # Fallback: daj równe score
            for d in drafts:
                d.total_score = 15

        return drafts

    def run(self, query: str, contexts: list[str]) -> SpeculativeResult:
        """Uruchom Speculative RAG.

        Args:
            query: Pytanie użytkownika.
            contexts: Lista kontekstów z retrieval.

        Returns:
            SpeculativeResult z draftami i wybranym najlepszym.
        """
        result = SpeculativeResult(query=query)

        # Podziel konteksty na N podzbiorów (nakładające się)
        n = len(contexts)
        subsets = []
        chunk_size = max(2, n // self._n_drafts + 1)

        for i in range(self._n_drafts):
            start = i * (n // self._n_drafts)
            subset = contexts[start:start + chunk_size]
            if not subset:
                subset = contexts[:chunk_size]
            subsets.append(subset)

        # Generuj drafty (Haiku)
        for i, subset in enumerate(subsets):
            text = self._generate_draft(query, subset)
            draft = Draft(
                draft_id=i + 1,
                text=text,
                context_subset=subset,
            )
            result.drafts.append(draft)

        # Weryfikuj drafty (Sonnet)
        result.drafts = self._verify_drafts(query, result.drafts)

        # Wybierz najlepszy
        best = max(result.drafts, key=lambda d: d.total_score)
        best.is_selected = True
        result.selected_draft = best
        result.final_answer = best.text

        return result

    def close(self):
        self._client.close()
