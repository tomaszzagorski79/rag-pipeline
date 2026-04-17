"""Klasyfikator złożoności zapytań dla Adaptive RAG.

Claude ocenia pytanie i klasyfikuje je jako simple/medium/complex,
co determinuje strategię retrieval.
"""

import anthropic

from config.settings import get_claude_config

CLASSIFY_PROMPT = """Sklasyfikuj poniższe pytanie pod kątem złożoności wyszukiwania.
Odpowiedz JEDNYM słowem: NO_RETRIEVAL, SIMPLE, MEDIUM lub COMPLEX.

NO_RETRIEVAL — ogólna wiedza, definicje, pytania na które LLM odpowie bez bazy wiedzy.
  Przykłady: "Co to jest SEO?", "Wyjaśnij czym jest e-commerce", "Co to jest API?"

SIMPLE — proste pytanie faktograficzne, jedno pojęcie, krótka odpowiedź.
  Przykłady: "Jaka stawka VAT w Niemczech?", "Co to jest cross-border?"

MEDIUM — pytanie wymagające kilku fragmentów kontekstu, porównanie, lista.
  Przykłady: "Jakie metody płatności stosować w cross-border?", "Jak zarejestrować VAT OSS?"

COMPLEX — pytanie wymagające syntezy wielu źródeł, analiza, wieloaspektowe.
  Przykłady: "Jak zacząć sprzedaż zagraniczną krok po kroku?", "Porównaj strategie wejścia na rynki UE"

Pytanie: {query}

Klasyfikacja:"""


class QueryClassifier:
    """Klasyfikator złożoności zapytań (Claude)."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def classify(self, query: str) -> str:
        """Sklasyfikuj pytanie jako SIMPLE, MEDIUM lub COMPLEX.

        Args:
            query: Pytanie użytkownika.

        Returns:
            'simple', 'medium' lub 'complex'.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=10,
            messages=[
                {"role": "user", "content": CLASSIFY_PROMPT.format(query=query)},
            ],
        )
        raw = response.content[0].text.strip().upper()

        if "NO_RETRIEVAL" in raw or "NO RETRIEVAL" in raw:
            return "no_retrieval"
        elif "SIMPLE" in raw:
            return "simple"
        elif "COMPLEX" in raw:
            return "complex"
        return "medium"

    def close(self):
        self._client.close()
