"""PageIndex — nawigator po drzewie (query-time reasoning).

Zamiast similarity search, Claude "rozumuje" po drzewie sekcji:
- Dostaje drzewo (tytuły + streszczenia, BEZ pełnego tekstu)
- Decyduje które node'y są relevantne dla pytania
- System pobiera pełny tekst TYLKO z wybranych
- Claude generuje odpowiedź
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config
from src.pageindex.tree_builder import TreeNode

NAVIGATE_PROMPT = """Jesteś ekspertem RAG bez użycia embeddingów — używasz rozumowania.

Otrzymujesz drzewo sekcji artykułu (tytuły + streszczenia).
Twoje zadanie: wybrać te node_id których PEŁNY tekst będzie potrzebny do odpowiedzi na pytanie.

Drzewo:
{tree}

Pytanie użytkownika: {query}

Zwróć JSON array z node_id najbardziej relevantnych sekcji (3-5 maksymalnie):
["node_id_1", "node_id_2", ...]

Zwróć TYLKO array, bez komentarzy."""

ANSWER_PROMPT = """Jesteś ekspertem e-commerce. Odpowiedz na pytanie na podstawie dostarczonych sekcji artykułu.

Sekcje:
{sections}

Pytanie: {query}

Odpowiedz po polsku, zwięźle, cytując sekcje jako [{{tytul_sekcji}}]."""


@dataclass
class PageIndexResult:
    """Wynik zapytania PageIndex."""

    query: str
    selected_nodes: list[TreeNode] = field(default_factory=list)
    reasoning_trace: str = ""
    answer: str = ""


class TreeNavigator:
    """Nawigator po drzewie PageIndex — reasoning-based retrieval."""

    def __init__(self):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model

    def _find_node(self, root: TreeNode, node_id: str) -> TreeNode | None:
        """Znajdź node po ID (rekurencyjnie)."""
        if root.node_id == node_id:
            return root
        for child in root.children:
            found = self._find_node(child, node_id)
            if found:
                return found
        return None

    def navigate(self, tree: TreeNode, query: str) -> PageIndexResult:
        """Rozumuj po drzewie i zwróć relevantne sekcje + odpowiedź.

        Args:
            tree: Root drzewa PageIndex.
            query: Pytanie użytkownika.

        Returns:
            PageIndexResult z wybranymi node'ami i odpowiedzią.
        """
        result = PageIndexResult(query=query)

        # 1. Claude wybiera node_ids (BEZ pełnego tekstu)
        tree_dict = tree.to_dict(include_content=False)
        tree_json = json.dumps(tree_dict, ensure_ascii=False, indent=2)

        nav_response = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": NAVIGATE_PROMPT.format(
                        tree=tree_json[:15000],  # ogranicz
                        query=query,
                    ),
                },
            ],
        )

        raw = nav_response.content[0].text.strip()
        result.reasoning_trace = raw

        # Parsuj JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            node_ids = json.loads(raw)
        except json.JSONDecodeError:
            node_ids = []

        # 2. Zbierz pełny tekst wybranych node'ów
        selected = []
        for nid in node_ids:
            node = self._find_node(tree, nid)
            if node:
                selected.append(node)

        result.selected_nodes = selected

        if not selected:
            result.answer = "PageIndex nie znalazł relevantnych sekcji w drzewie."
            return result

        # 3. Generuj odpowiedź z pełnego tekstu
        sections_text = "\n\n".join(
            f"## {n.title}\n{n.content}" for n in selected if n.content
        )

        answer_response = self._client.messages.create(
            model=self._model,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        sections=sections_text[:12000],
                        query=query,
                    ),
                },
            ],
        )
        result.answer = answer_response.content[0].text

        return result

    def close(self):
        self._client.close()
