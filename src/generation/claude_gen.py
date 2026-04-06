"""Moduł generowania odpowiedzi przez Claude API.

RAG generation: kontekst z retrieval + pytanie użytkownika → odpowiedź Claude.
"""

import anthropic

from config.settings import ClaudeConfig, get_claude_config

# System prompt instruujący Claude do odpowiadania na podstawie kontekstu
SYSTEM_PROMPT = """Jesteś ekspertem e-commerce i SEO. Odpowiadasz na pytania WYŁĄCZNIE na podstawie
dostarczonych fragmentów artykułów. Przestrzegaj następujących zasad:

1. Odpowiadaj po polsku.
2. Bazuj TYLKO na dostarczonym kontekście — nie dodawaj wiedzy spoza niego.
3. Jeśli kontekst nie zawiera odpowiedzi na pytanie, powiedz wprost:
   "Na podstawie dostępnych fragmentów nie mogę odpowiedzieć na to pytanie."
4. Cytuj lub parafrazuj fragmenty kontekstu, wskazując numer źródła [1], [2] itd.
5. Bądź konkretny i zwięzły — unikaj ogólników.
"""


class ClaudeGenerator:
    """Generator odpowiedzi RAG oparty na Claude API."""

    def __init__(self, config: ClaudeConfig | None = None):
        cfg = config or get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model
        self._max_tokens = cfg.max_tokens

    def generate(
        self,
        query: str,
        contexts: list[str],
        system_prompt: str | None = None,
    ) -> str:
        """Wygeneruj odpowiedź na podstawie kontekstu i pytania.

        Args:
            query: Pytanie użytkownika.
            contexts: Lista fragmentów tekstu (top-K z retrieval).
            system_prompt: Opcjonalny custom system prompt.

        Returns:
            Tekst odpowiedzi.
        """
        # Formatuj kontekst jako numerowane fragmenty
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(f"[{i}] {ctx}")
        formatted_context = "\n\n".join(context_parts)

        user_message = (
            f"KONTEKST (fragmenty artykułów):\n"
            f"{formatted_context}\n\n"
            f"---\n\n"
            f"PYTANIE: {query}"
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt or SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )

        return response.content[0].text

    def close(self):
        """Zamknij klienta."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
