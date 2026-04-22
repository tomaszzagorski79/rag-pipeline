"""Skrypt do pobierania artykułów ze stron WWW przez Jina Reader.

Użycie:
    python scripts/01_scrape_articles.py                     # pobierz wszystkie z article_urls.txt
    python scripts/01_scrape_articles.py --url "https://..."  # pobierz pojedynczy artykuł
    python scripts/01_scrape_articles.py --dry-run             # pokaż co zostanie pobrane
"""

import argparse
import sys
from pathlib import Path

# Dodaj katalog główny projektu do ścieżki
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.scraper.jina_reader import (
    pobierz_artykul,
    pobierz_wszystkie,
    wczytaj_liste_url,
    zapisz_artykul,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Pobieranie artykułów przez Jina Reader")
    parser.add_argument("--url", type=str, help="Pojedynczy URL do pobrania")
    parser.add_argument("--dry-run", action="store_true", help="Tylko pokaż listę URL-i")
    parser.add_argument(
        "--delay", type=float, default=3.0, help="Opóźnienie między requestami (s)"
    )
    args = parser.parse_args()

    if args.url:
        # Tryb pojedynczego URL-a
        console.print(f"[bold]Pobieranie: {args.url}[/bold]")
        artykul = pobierz_artykul(args.url)
        plik = zapisz_artykul(artykul)
        console.print(f"[green]✓ Zapisano: {plik}[/green]")
        console.print(f"  Tytuł: {artykul['tytul']}")
        console.print(f"  Długość: {len(artykul['markdown'])} znaków")

    elif args.dry_run:
        # Tryb podglądu
        urls = wczytaj_liste_url()
        if not urls:
            console.print("[yellow]Brak URL-i w article_urls.txt[/yellow]")
            return
        console.print(f"[bold]Znaleziono {len(urls)} URL-i:[/bold]")
        for i, url in enumerate(urls, 1):
            console.print(f"  {i}. {url}")

    else:
        # Tryb masowy
        pobierz_wszystkie(opoznienie=args.delay)


if __name__ == "__main__":
    main()
