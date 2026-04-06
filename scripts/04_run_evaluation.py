"""Skrypt do uruchomienia ewaluacji RAGAS na zestawie testowym.

Użycie:
    python scripts/04_run_evaluation.py --method all
    python scripts/04_run_evaluation.py --method naive header
    python scripts/04_run_evaluation.py --method all --limit 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.chunking.registry import AVAILABLE_METHODS
from src.evaluation.ragas_eval import run_evaluation, zapisz_wyniki

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja RAGAS pipeline'u RAG")
    parser.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        help="Metody chunkingu do ewaluacji: naive, header, semantic, all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Liczba kontekstów (top-K) na pytanie",
    )
    args = parser.parse_args()

    metody = AVAILABLE_METHODS if "all" in args.method else args.method

    console.print("[bold]Uruchamianie ewaluacji RAGAS...[/bold]\n")
    wyniki = run_evaluation(metody, limit=args.limit)

    if wyniki:
        zapisz_wyniki(wyniki)
    else:
        console.print("[yellow]Brak wyników do zapisania.[/yellow]")


if __name__ == "__main__":
    main()
