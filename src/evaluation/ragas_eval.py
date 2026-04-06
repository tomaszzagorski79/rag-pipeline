"""Moduł ewaluacji pipeline'u RAG za pomocą RAGAS.

Mierzy: faithfulness, answer_relevancy, context_precision.
Obsługuje Claude jako LLM evaluator (przez langchain-anthropic).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from config.settings import get_claude_config, get_paths, get_qdrant_config
from src.generation.claude_gen import ClaudeGenerator
from src.retrieval.hybrid_search import HybridRetriever

console = Console()


@dataclass
class TestQuestion:
    """Pytanie testowe z oczekiwaną odpowiedzią."""

    question: str
    ground_truth: str
    source_article: str = ""


def wczytaj_test_set(sciezka: Path | None = None) -> list[TestQuestion]:
    """Wczytaj zestaw testowy z pliku JSON.

    Format pliku:
    [
        {
            "question": "Jakie są metody zwiększania konwersji?",
            "ground_truth": "Główne metody to...",
            "source_article": "artykul.md"
        }
    ]
    """
    plik = sciezka or get_paths().test_set_file

    if not plik.exists():
        console.print(f"[red]Brak pliku testowego: {plik}[/red]")
        console.print("[yellow]Utwórz data/test_set.json z pytaniami i oczekiwanymi odpowiedziami.[/yellow]")
        return []

    data = json.loads(plik.read_text(encoding="utf-8"))
    return [TestQuestion(**item) for item in data]


def run_evaluation(
    metody: list[str],
    test_questions: list[TestQuestion] | None = None,
    limit: int = 5,
) -> dict:
    """Uruchom ewaluację RAGAS dla wybranych metod chunkingu.

    Args:
        metody: Lista metod chunkingu do ewaluacji.
        test_questions: Pytania testowe (domyślnie z test_set.json).
        limit: Liczba kontekstów na pytanie (top-K).

    Returns:
        dict z wynikami per metoda:
        {
            "naive": {"faithfulness": 0.85, "answer_relevancy": 0.72, ...},
            ...
        }
    """
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        ResponseRelevancy,
    )

    from src.embeddings.jina_embed import JinaEmbeddingsLangChain

    # Załaduj pytania testowe
    questions = test_questions or wczytaj_test_set()
    if not questions:
        return {}

    console.print(f"[bold]Pytań testowych: {len(questions)}[/bold]")

    # LLM evaluator (Claude)
    claude_cfg = get_claude_config()
    evaluator_llm = ChatAnthropic(
        model=claude_cfg.model,
        anthropic_api_key=claude_cfg.api_key,
    )
    evaluator_embeddings = JinaEmbeddingsLangChain()

    # Retriever i generator
    retriever = HybridRetriever()
    generator = ClaudeGenerator()
    qdrant_cfg = get_qdrant_config()

    all_results = {}

    for metoda in metody:
        collection_name = qdrant_cfg.collection_name(metoda)

        if not retriever.store.collection_exists(collection_name):
            console.print(f"[yellow]Kolekcja '{collection_name}' nie istnieje — pomijam.[/yellow]")
            continue

        console.print(f"\n[bold cyan]═══ Ewaluacja: {metoda.upper()} ═══[/bold cyan]")

        # Zbierz sample'e
        samples = []
        detailed_results = []

        for q in questions:
            console.print(f"  ❓ {q.question[:60]}...")

            # Retrieval
            search_results = retriever.search(q.question, collection_name, limit=limit)
            contexts = [r.text for r in search_results]

            # Generation
            answer = generator.generate(q.question, contexts)

            # RAGAS sample
            sample = SingleTurnSample(
                user_input=q.question,
                response=answer,
                retrieved_contexts=contexts,
                reference=q.ground_truth,
            )
            samples.append(sample)

            # Zachowaj szczegóły
            detailed_results.append({
                "question": q.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": q.ground_truth,
                "source_article": q.source_article,
            })

        # Ewaluacja RAGAS
        console.print("\n  [bold]Ewaluacja RAGAS...[/bold]")
        dataset = EvaluationDataset(samples=samples)

        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
        ]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            )

            scores = result.to_pandas().mean(numeric_only=True).to_dict()
            all_results[metoda] = {
                "scores": scores,
                "detailed": detailed_results,
                "per_question": result.to_pandas().to_dict(orient="records"),
            }

            # Wyświetl wyniki
            for metric_name, score in scores.items():
                console.print(f"    {metric_name}: {score:.4f}")

        except Exception as e:
            console.print(f"  [red]Błąd RAGAS: {e}[/red]")
            all_results[metoda] = {
                "scores": {},
                "detailed": detailed_results,
                "error": str(e),
            }

    return all_results


def zapisz_wyniki(wyniki: dict, output_dir: Path | None = None) -> Path:
    """Zapisz wyniki ewaluacji do pliku JSON.

    Args:
        wyniki: dict z wynikami z run_evaluation().
        output_dir: Katalog docelowy (domyślnie data/results/).

    Returns:
        Ścieżka do zapisanego pliku.
    """
    katalog = output_dir or get_paths().results
    katalog.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plik = katalog / f"evaluation_{timestamp}.json"

    # Zapisz (bez detailed kontekstów — za duże)
    wyniki_do_zapisu = {}
    for metoda, data in wyniki.items():
        wyniki_do_zapisu[metoda] = {
            "scores": data.get("scores", {}),
            "per_question": data.get("per_question", []),
            "error": data.get("error"),
        }

    plik.write_text(
        json.dumps(wyniki_do_zapisu, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"\n[green]✓ Wyniki zapisane: {plik}[/green]")
    return plik
