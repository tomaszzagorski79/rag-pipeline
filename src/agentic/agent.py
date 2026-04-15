"""Agentic RAG — agent Claude decyduje jakich narzędzi użyć.

Używa Claude tool_use API. Agent ma dostęp do:
- search_hybrid: hybrid search (BM25+vector+RRF)
- search_hyde: HyDE (hipotetyczna odpowiedź → search)
- rerank_results: re-ranking cross-encoderem
- reformulate_query: przeformułowanie pytania

ReACT loop: Thought → Action → Observation → powtórz → Final Answer.
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config
from src.retrieval.hybrid_search import HybridRetriever, SearchResult


AGENT_SYSTEM_PROMPT = """Jesteś inteligentnym agentem RAG. Odpowiadasz na pytania użytkownika,
wykorzystując narzędzia retrieval dostępne dla Ciebie.

ZASADY:
1. Zacznij od search_hybrid dla większości pytań.
2. Jeśli wyniki są słabe (niski score lub nie odpowiadają na pytanie) — użyj search_hyde.
3. Jeśli masz 5+ wyników ale chcesz lepszej precyzji — użyj rerank_results.
4. Jeśli pytanie jest niejasne — użyj reformulate_query.
5. Gdy masz wystarczający kontekst — odpowiedz użytkownikowi (bez wywoływania narzędzi).
6. Nie wywołuj więcej niż 4 narzędzi — optymalizuj koszty.

Odpowiadaj po polsku, cytuj źródła jako [1], [2]. Bazuj tylko na kontekście z narzędzi.
"""

# Definicje narzędzi dla Claude tool_use API
TOOLS = [
    {
        "name": "search_hybrid",
        "description": "Wyszukaj dokumenty hybrid searchem (BM25 + vector). Zwraca top-K najbardziej pasujących chunków.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Zapytanie do wyszukania"},
                "limit": {"type": "integer", "description": "Liczba wyników (1-20)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_hyde",
        "description": "Wyszukaj używając HyDE — generuje hipotetyczną odpowiedź i szuka po niej. Dobre dla złożonych pytań.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Zapytanie użytkownika"},
                "limit": {"type": "integer", "description": "Liczba wyników", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rerank_results",
        "description": "Re-rankuj wyniki cross-encoderem dla lepszej precyzji. Użyj gdy masz wyniki ale chcesz je przefiltrować.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Oryginalne zapytanie"},
                "top_k": {"type": "integer", "description": "Ile top wyników zostawić", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "reformulate_query",
        "description": "Przeformułuj pytanie gdy jest niejasne lub retrieval zwrócił słabe wyniki.",
        "input_schema": {
            "type": "object",
            "properties": {
                "original_query": {"type": "string", "description": "Oryginalne pytanie"},
            },
            "required": ["original_query"],
        },
    },
]


@dataclass
class AgentStep:
    """Pojedynczy krok agenta (tool call lub final answer)."""

    step_type: str  # "thought", "tool_call", "tool_result", "final_answer"
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_output: str = ""
    content: str = ""


@dataclass
class AgentResult:
    """Wynik działania agenta."""

    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_tool_calls: int = 0
    retrieved_chunks: list[SearchResult] = field(default_factory=list)


class AgenticRAG:
    """Agent RAG z tool_use — podejmuje decyzje samodzielnie."""

    def __init__(self, collection_name: str):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model
        self._collection = collection_name
        self._max_iterations = 5

        # Lazy init narzędzi
        self._retriever: HybridRetriever | None = None
        self._hyde = None
        self._reranker = None
        self._last_results: list[SearchResult] = []

    def _get_retriever(self):
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def _get_hyde(self):
        if self._hyde is None:
            from src.hyde.hyde_generator import HyDEGenerator
            self._hyde = HyDEGenerator()
        return self._hyde

    def _get_reranker(self):
        if self._reranker is None:
            from src.reranking.flashrank_reranker import FlashRankReranker
            self._reranker = FlashRankReranker()
        return self._reranker

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Wykonaj narzędzie i zwróć tekst wyników."""
        if tool_name == "search_hybrid":
            retriever = self._get_retriever()
            results = retriever.search(
                tool_input["query"],
                self._collection,
                limit=tool_input.get("limit", 5),
            )
            self._last_results = results
            return self._format_results(results)

        elif tool_name == "search_hyde":
            hyde = self._get_hyde()
            hypothesis, results = hyde.search_with_hyde(
                tool_input["query"],
                self._collection,
                limit=tool_input.get("limit", 5),
            )
            self._last_results = results
            return f"Hipoteza HyDE: {hypothesis[:300]}...\n\nWyniki:\n{self._format_results(results)}"

        elif tool_name == "rerank_results":
            if not self._last_results:
                return "Brak wyników do re-rankingu. Wywołaj search_hybrid lub search_hyde najpierw."
            reranker = self._get_reranker()
            reranked = reranker.rerank(
                tool_input["query"],
                self._last_results,
                top_k=tool_input.get("top_k", 5),
            )
            self._last_results = reranked
            return self._format_results(reranked)

        elif tool_name == "reformulate_query":
            response = self._client.messages.create(
                model=self._model,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": f"Przeformułuj pytanie używając innych słów kluczowych. Zwróć TYLKO nowe pytanie.\n\nOryginał: {tool_input['original_query']}",
                    },
                ],
            )
            return response.content[0].text.strip()

        return f"Nieznane narzędzie: {tool_name}"

    def _format_results(self, results: list[SearchResult]) -> str:
        """Formatuj wyniki dla agenta."""
        if not results:
            return "Brak wyników."
        parts = []
        for i, r in enumerate(results[:5], 1):
            parts.append(f"[{i}] (score: {r.score:.3f}) {r.text[:400]}")
        return "\n\n".join(parts)

    def run(self, query: str) -> AgentResult:
        """Uruchom agenta dla zapytania.

        Args:
            query: Pytanie użytkownika.

        Returns:
            AgentResult z pełnym łańcuchem kroków.
        """
        result = AgentResult(query=query)
        messages = [{"role": "user", "content": query}]

        for iteration in range(self._max_iterations):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=AGENT_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # Zachowaj całą odpowiedź w historii
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Sprawdź stop_reason
            if response.stop_reason == "end_turn":
                # Final answer
                text_parts = [b.text for b in assistant_content if hasattr(b, "text")]
                result.final_answer = "\n".join(text_parts)
                result.steps.append(AgentStep(
                    step_type="final_answer",
                    content=result.final_answer,
                ))
                break

            # Tool calls
            tool_results = []
            for block in assistant_content:
                if block.type == "text" and block.text.strip():
                    result.steps.append(AgentStep(
                        step_type="thought",
                        content=block.text,
                    ))
                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    result.total_tool_calls += 1

                    # Wykonaj narzędzie
                    tool_output = self._execute_tool(tool_name, tool_input)

                    result.steps.append(AgentStep(
                        step_type="tool_call",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_output=tool_output[:1000],
                    ))

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_output,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        # Dołącz ostatnie retrieved chunks
        result.retrieved_chunks = self._last_results

        return result

    def close(self):
        self._client.close()
        if self._retriever:
            self._retriever.store.close()
