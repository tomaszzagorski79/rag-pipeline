"""Interaktywne demo zapytań do pipeline'u RAG.

Użycie:
    python scripts/03_query.py                          # wszystkie kolekcje
    python scripts/03_query.py --method naive header     # wybrane metody
    python scripts/03_query.py --limit 3                 # top-3 kontekstów
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from config.settings import get_qdrant_config
from src.chunking.registry import AVAILABLE_METHODS
from src.generation.claude_gen import ClaudeGenerator
from src.retrieval.hybrid_search import HybridRetriever

console = Console()


def interactive_query(metody: list[str], limit: int = 5):
    """Interaktywna pętla zapytań — pytaj i porównuj odpowiedzi."""
    qdrant_cfg = get_qdrant_config()
    retriever = HybridRetriever()
    generator = ClaudeGenerator()

    # Sprawdź które kolekcje istnieją
    dostepne = []
    for m in metody:
        name = qdrant_cfg.collection_name(m)
        if retriever.store.collection_exists(name):
            dostepne.append(m)
        else:
            console.print(f"[yellow]Kolekcja '{name}' nie istnieje — pomijam {m}[/yellow]")

    if not dostepne:
        console.print("[red]Brak dostępnych kolekcji. Uruchom najpierw 02_chunk_and_index.py[/red]")
        return

    console.print(f"[bold green]Dostępne metody: {', '.join(dostepne)}[/bold green]")
    console.print("[dim]Wpisz pytanie (lub 'q' aby zakończyć)[/dim]\n")

    while True:
        query = console.input("[bold]❓ Pytanie: [/bold]").strip()
        if query.lower() in ("q", "quit", "exit", ""):
            break

        for metoda in dostepne:
            collection_name = qdrant_cfg.collection_name(metoda)
            console.print(f"\n[bold cyan]── {metoda.upper()} ──[/bold cyan]")

            # Retrieval
            results = retriever.search(query, collection_name, limit=limit)

            if not results:
                console.print("  [yellow]Brak wyników.[/yellow]")
                continue

            # Pokaż konteksty
            for i, r in enumerate(results, 1):
                preview = r.text[:150].replace("\n", " ")
                console.print(f"  [{i}] (score: {r.score:.4f}) {preview}...")

            # Generation
            contexts = [r.text for r in results]
            console.print("  [dim]Generowanie odpowiedzi...[/dim]")
            answer = generator.generate(query, contexts)

            console.print(Panel(
                Markdown(answer),
                title=f"Odpowiedź ({metoda})",
                border_style="green",
            ))

        console.print()


def main():
    parser = argparse.ArgumentParser(description="Interaktywne demo zapytań RAG")
    parser.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        help="Metody: naive, header, semantic, all",
    )
    parser.add_argument("--limit", type=int, default=5, help="Liczba kontekstów (top-K)")
    args = parser.parse_args()

    metody = AVAILABLE_METHODS if "all" in args.method else args.method
    interactive_query(metody, limit=args.limit)


if __name__ == "__main__":
    main()
