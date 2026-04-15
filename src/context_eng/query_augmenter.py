"""Query Augmentation — przygotowanie zapytania przed retrieval.

Komponent 2 Context Engineering:
- Query rewriting (przeformułowanie dla jasności)
- Query fan-out (dekompozycja na sub-zapytania)
- Query intent extraction (klasyfikacja typu pytania)
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config

REWRITE_PROMPT = """Przeformułuj pytanie użytkownika tak żeby było bardziej precyzyjne
dla wyszukiwania w bazie wiedzy. Zachowaj intencję, ale użyj terminologii branżowej
i słów kluczowych które mogą pasować do treści.

Oryginał: {query}

Zwróć TYLKO przeformułowane pytanie, bez komentarzy."""

FANOUT_PROMPT = """Rozbij pytanie użytkownika na {n} sub-zapytań które razem pokryją temat.
Każde sub-zapytanie powinno być samowystarczalne i skupiać się na innym aspekcie.

Pytanie: {query}

Zwróć JSON array z sub-zapytaniami:
["sub-zapytanie 1", "sub-zapytanie 2", ...]

Zwróć TYLKO JSON array."""

INTENT_PROMPT = """Sklasyfikuj typ pytania użytkownika. Wybierz JEDEN typ:

- FACTUAL: prosty fakt (np. "jaka stawka VAT w Niemczech")
- COMPARATIVE: porównanie dwóch+ rzeczy (np. "co lepsze: OSS czy IOSS")
- PROCEDURAL: jak coś zrobić (np. "jak zarejestrować VAT OSS")
- EXPLORATORY: otwarte eksploracyjne (np. "co warto wiedzieć o cross-border")
- ANALYTICAL: analiza/synteza (np. "porównaj strategie wejścia na rynki UE")

Pytanie: {query}

Zwróć TYLKO jeden typ (wielkimi literami)."""


@dataclass
class AugmentedQuery:
    """Wynik query augmentation."""

    original: str
    rewritten: str = ""
    sub_queries: list[str] = field(default_factory=list)
    intent: str = "FACTUAL"
    reasoning: str = ""


class QueryAugmenter:
    """Augmentacja zapytań — rewriting, fan-out, klasyfikacja intencji."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def _call(self, prompt: str, max_tokens: int = 500) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def rewrite(self, query: str) -> str:
        """Przeformułuj pytanie dla lepszego retrieval."""
        return self._call(REWRITE_PROMPT.format(query=query), max_tokens=200)

    def fanout(self, query: str, n: int = 3) -> list[str]:
        """Rozbij pytanie na sub-zapytania."""
        raw = self._call(FANOUT_PROMPT.format(query=query, n=n), max_tokens=400)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [str(q) for q in result]
        except json.JSONDecodeError:
            pass
        return [query]

    def classify_intent(self, query: str) -> str:
        """Sklasyfikuj typ pytania."""
        raw = self._call(INTENT_PROMPT.format(query=query), max_tokens=20).upper()
        for intent in ["FACTUAL", "COMPARATIVE", "PROCEDURAL", "EXPLORATORY", "ANALYTICAL"]:
            if intent in raw:
                return intent
        return "FACTUAL"

    def augment(
        self,
        query: str,
        do_rewrite: bool = True,
        do_fanout: bool = True,
        fanout_n: int = 3,
    ) -> AugmentedQuery:
        """Pełna augmentacja — klasyfikacja + rewriting + fanout.

        Args:
            query: Oryginalne pytanie.
            do_rewrite: Czy przeformułować.
            do_fanout: Czy rozbić na sub-zapytania.
            fanout_n: Liczba sub-zapytań.

        Returns:
            AugmentedQuery z wszystkimi wariantami.
        """
        result = AugmentedQuery(original=query)

        # Klasyfikacja intencji
        result.intent = self.classify_intent(query)

        # Rewriting
        if do_rewrite:
            result.rewritten = self.rewrite(query)
        else:
            result.rewritten = query

        # Fan-out (tylko dla złożonych pytań)
        if do_fanout and result.intent in ("COMPARATIVE", "EXPLORATORY", "ANALYTICAL"):
            result.sub_queries = self.fanout(query, n=fanout_n)
        else:
            result.sub_queries = [result.rewritten]

        result.reasoning = (
            f"Intencja: {result.intent}. "
            f"{'Rozbite na ' + str(len(result.sub_queries)) + ' sub-zapytań.' if len(result.sub_queries) > 1 else 'Brak fan-out.'}"
        )

        return result

    def close(self):
        self._client.close()
