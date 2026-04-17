"""FLARE — Forward-Looking Active Retrieval.

Retrieval nie jest jednorazowy przed generacją — odbywa się ITERACYJNIE
w trakcie generacji, gdy model jest niepewny.

Uproszczona wersja (Claude API nie daje logprobs):
- Generuj zdanie po zdaniu
- Claude ocenia swoją pewność per zdanie
- Jeśli niska → retrieval z tym zdaniem jako query → regeneruj
"""

import json
from dataclasses import dataclass, field

import anthropic

from config.settings import get_claude_config
from src.retrieval.hybrid_search import HybridRetriever, SearchResult

GENERATE_SENTENCE_PROMPT = """Kontynuuj odpowiedź na pytanie. Napisz JEDNO następne zdanie.
Bazuj na dostarczonym kontekście. Jeśli nie masz informacji, napisz "[NIEPEWNY]" na końcu zdania.

Pytanie: {query}
Kontekst: {context}
Dotychczasowa odpowiedź: {partial_answer}

Następne zdanie (JEDNO, po polsku):"""

CONFIDENCE_PROMPT = """Oceń pewność poniższego zdania na skali 1-10.
1 = kompletnie wymyślone, 10 = w pełni poparte kontekstem.

Zdanie: {sentence}
Kontekst: {context}

Zwróć TYLKO liczbę (1-10):"""


@dataclass
class FLAREStep:
    """Pojedynczy krok FLARE."""

    sentence: str
    confidence: int
    needed_retrieval: bool = False
    retrieval_query: str = ""
    new_contexts: list[str] = field(default_factory=list)
    regenerated: bool = False


@dataclass
class FLAREResult:
    """Wynik FLARE."""

    query: str
    steps: list[FLAREStep] = field(default_factory=list)
    final_answer: str = ""
    total_retrievals: int = 0
    initial_contexts: list[str] = field(default_factory=list)


class FLAREGenerator:
    """FLARE — generacja z aktywnym retrieval w trakcie."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        confidence_threshold: int = 5,
        max_sentences: int = 10,
    ):
        cfg = get_claude_config()
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._model = cfg.model
        self._retriever = retriever or HybridRetriever()
        self._threshold = confidence_threshold
        self._max_sentences = max_sentences

    def _generate_sentence(self, query: str, context: str, partial: str) -> str:
        """Wygeneruj jedno zdanie kontynuacji."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": GENERATE_SENTENCE_PROMPT.format(
                    query=query, context=context[:8000], partial_answer=partial or "(brak)",
                ),
            }],
        )
        return response.content[0].text.strip()

    def _assess_confidence(self, sentence: str, context: str) -> int:
        """Oceń pewność zdania (1-10)."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": CONFIDENCE_PROMPT.format(sentence=sentence, context=context[:5000]),
            }],
        )
        raw = response.content[0].text.strip()
        try:
            return max(1, min(10, int(raw.split()[0])))
        except (ValueError, IndexError):
            return 5

    def run(
        self,
        query: str,
        collection_name: str,
        initial_limit: int = 5,
    ) -> FLAREResult:
        """Uruchom FLARE — generacja zdanie po zdaniu z aktywnym retrieval.

        Args:
            query: Pytanie użytkownika.
            collection_name: Nazwa kolekcji Qdrant.
            initial_limit: Początkowe top-K kontekstów.

        Returns:
            FLAREResult z krokami i finalną odpowiedzią.
        """
        result = FLAREResult(query=query)

        # Początkowy retrieval
        initial_results = self._retriever.search(query, collection_name, limit=initial_limit)
        contexts = [r.text for r in initial_results]
        result.initial_contexts = contexts.copy()

        partial_answer = ""

        for _ in range(self._max_sentences):
            context_str = "\n\n".join(contexts[:10])

            # Generuj zdanie
            sentence = self._generate_sentence(query, context_str, partial_answer)
            if not sentence or sentence.startswith("[KONIEC]"):
                break

            # Oceń pewność
            confidence = self._assess_confidence(sentence, context_str)

            step = FLAREStep(sentence=sentence, confidence=confidence)

            # Jeśli niska pewność → active retrieval
            if confidence < self._threshold or "[NIEPEWNY]" in sentence:
                step.needed_retrieval = True
                step.retrieval_query = sentence.replace("[NIEPEWNY]", "").strip()

                # Retrieval z bieżącym zdaniem jako query
                new_results = self._retriever.search(
                    step.retrieval_query, collection_name, limit=3
                )
                new_contexts = [r.text for r in new_results]
                step.new_contexts = new_contexts
                contexts.extend(new_contexts)  # Wzbogać kontekst
                result.total_retrievals += 1

                # Regeneruj zdanie z nowym kontekstem
                context_str = "\n\n".join(contexts[:12])
                regenerated = self._generate_sentence(query, context_str, partial_answer)
                if regenerated:
                    sentence = regenerated
                    step.sentence = regenerated
                    step.regenerated = True
                    step.confidence = self._assess_confidence(regenerated, context_str)

            # Wyczyść markery
            sentence = sentence.replace("[NIEPEWNY]", "").strip()
            step.sentence = sentence

            result.steps.append(step)
            partial_answer += " " + sentence

        result.final_answer = partial_answer.strip()
        return result

    def close(self):
        self._client.close()
        self._retriever.store.close()
