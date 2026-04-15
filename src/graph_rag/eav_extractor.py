"""Ekstraktor EAV — Claude wyciąga trójki (Entity, Attribute, Value) z chunków.

Używany do budowania grafu wiedzy z treści artykułów.
"""

import json

import anthropic

from config.settings import get_claude_config

EXTRACT_EAV_PROMPT = """Wyciągnij wszystkie fakty z poniższego tekstu jako trójki EAV (Entity-Attribute-Value).

EAV to podstawowy model danych:
- Entity: encja (marka, produkt, kraj, osoba, pojęcie)
- Attribute: atrybut tej encji (stawka, cena, data, nazwa, cecha)
- Value: wartość atrybutu

Przykłady:
- "VAT w Niemczech to 19%" → {{"entity": "VAT Niemcy", "attribute": "stawka", "value": "19%"}}
- "Amazon ma siedzibę w Seattle" → {{"entity": "Amazon", "attribute": "siedziba", "value": "Seattle"}}
- "Próg VAT OSS to 10000 EUR" → {{"entity": "VAT OSS", "attribute": "próg", "value": "10000 EUR"}}

Tekst:
{text}

Zwróć JSON array trójek (max 10, tylko najważniejsze fakty):
[{{"entity": "...", "attribute": "...", "value": "..."}}]

Zwróć TYLKO JSON, bez komentarzy."""


class EAVExtractor:
    """Ekstraktor EAV oparty na Claude."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def extract(self, text: str) -> list[dict]:
        """Wyciągnij trójki EAV z tekstu.

        Args:
            text: Treść chunka/sekcji.

        Returns:
            Lista dict z kluczami entity, attribute, value.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_EAV_PROMPT.format(text=text[:3000]),
                },
            ],
        )
        raw = response.content[0].text.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            triples = json.loads(raw)
            if isinstance(triples, list):
                return [t for t in triples if isinstance(t, dict) and all(k in t for k in ("entity", "attribute", "value"))]
        except json.JSONDecodeError:
            pass

        return []

    def close(self):
        self._client.close()
