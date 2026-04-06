"""Scraper artykułów przez Jina Reader API.

Konwertuje URL artykułu na czysty markdown z zachowaną strukturą H2/H3.
"""

import re
import time
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console

from config.settings import get_jina_config, get_paths

console = Console()


def _wyciagnij_slug(url: str) -> str:
    """Wyciąga slug z URL-a do nazwy pliku.

    Przykład: 'https://www.idosell.com/pl/blog/moj-artykul-123' -> 'moj-artykul-123'
    """
    # Usuń trailing slash i weź ostatni segment
    sciezka = url.rstrip("/").split("/")[-1]
    # Wyczyść znaki specjalne
    slug = re.sub(r"[^\w\-]", "_", sciezka)
    return slug[:100]  # ogranicz długość


def _wyczysc_nawigacje(markdown: str) -> str:
    """Usuwa nawigację/menu/footer — zostawia tylko treść artykułu.

    Szuka pierwszego nagłówka H2 (##) jako początku treści.
    Ucina footer od typowych wzorców końca artykułu.
    """
    lines = markdown.split("\n")

    # Znajdź pierwszy H2 — tam zaczyna się treść artykułu
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("## "):
            start = i
            break

    # Znajdź koniec treści — szukaj wzorców footera
    end = len(lines)
    footer_markers = [
        "## Powiązane artykuły",
        "## Podobne artykuły",
        "## Najczęściej czytane",
        "## Czytaj również",
        "## Zapisz się",
        "## Newsletter",
        "Wszystkie prawa zastrzeżone",
        "© IdoSell",
        "Polityka prywatności",
        "### IdoSell",
        "### Dokumenty i regulaminy",
        "### Kontakt",
        "[O nas]",
        "[Kariera]",
        "Dowiedz się więcej o IdoSell",
        "Porozmawiaj z ekspertem",
        "## Inne artykuły",
        "IdoSell Newsletter",
        "Wypełnij formularz",
        "Zapisz się do newslettera",
        "Chcesz być na bieżąco",
        "Chcesz wejść na nowe rynki",
    ]
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if any(marker in stripped for marker in footer_markers):
            end = i
            break

    # Wyciągnij tytuł (H1) z oryginalnego tekstu jeśli jest przed H2
    tytul_line = ""
    for line in lines[:start]:
        if line.strip().startswith("# ") and not line.strip().startswith("## "):
            tytul_line = line + "\n\n"
            break

    content = "\n".join(lines[start:end]).strip()
    return tytul_line + content


def _wyciagnij_tytul(markdown: str) -> str:
    """Wyciąga tytuł (pierwszy H1) z markdownu."""
    for linia in markdown.split("\n"):
        if linia.startswith("# "):
            return linia.lstrip("# ").strip()
    return "Bez tytułu"


def pobierz_artykul(url: str, jina_config=None) -> dict:
    """Pobiera artykuł z URL-a przez Jina Reader API.

    Args:
        url: URL artykułu do pobrania.
        jina_config: Opcjonalna konfiguracja Jina (domyślnie z .env).

    Returns:
        dict z kluczami: 'markdown', 'tytul', 'url', 'slug'

    Raises:
        httpx.HTTPStatusError: Gdy API zwróci błąd.
    """
    config = jina_config or get_jina_config()

    reader_url = f"{config.reader_url}{url}"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "X-Return-Format": "markdown",
        "Accept": "text/markdown",
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(reader_url, headers=headers)
        response.raise_for_status()

    markdown = response.text
    markdown = _wyczysc_nawigacje(markdown)
    slug = _wyciagnij_slug(url)
    tytul = _wyciagnij_tytul(markdown)

    return {
        "markdown": markdown,
        "tytul": tytul,
        "url": url,
        "slug": slug,
    }


def zapisz_artykul(artykul: dict, katalog_wyjsciowy: Path | None = None) -> Path:
    """Zapisuje artykuł jako plik markdown z YAML frontmatter.

    Args:
        artykul: dict z kluczami 'markdown', 'tytul', 'url', 'slug'.
        katalog_wyjsciowy: Katalog docelowy (domyślnie data/raw/).

    Returns:
        Ścieżka do zapisanego pliku.
    """
    katalog = katalog_wyjsciowy or get_paths().raw_articles
    katalog.mkdir(parents=True, exist_ok=True)

    plik = katalog / f"{artykul['slug']}.md"

    frontmatter = (
        f"---\n"
        f"title: \"{artykul['tytul']}\"\n"
        f"source_url: \"{artykul['url']}\"\n"
        f"scraped_at: \"{datetime.now().isoformat()}\"\n"
        f"---\n\n"
    )

    plik.write_text(frontmatter + artykul["markdown"], encoding="utf-8")
    return plik


def wczytaj_liste_url(sciezka: Path | None = None) -> list[str]:
    """Wczytuje listę URL-i z pliku (1 URL na linię, # = komentarz).

    Args:
        sciezka: Ścieżka do pliku z URL-ami (domyślnie data/article_urls.txt).

    Returns:
        Lista URL-i do pobrania.
    """
    plik = sciezka or get_paths().article_urls_file

    if not plik.exists():
        console.print(f"[red]Plik nie istnieje: {plik}[/red]")
        return []

    urls = []
    for linia in plik.read_text(encoding="utf-8").splitlines():
        linia = linia.strip()
        if linia and not linia.startswith("#"):
            urls.append(linia)

    return urls


def pobierz_wszystkie(
    urls: list[str] | None = None,
    katalog_wyjsciowy: Path | None = None,
    opoznienie: float = 3.0,
) -> list[Path]:
    """Pobiera listę artykułów z rate limitingiem.

    Args:
        urls: Lista URL-i (domyślnie z article_urls.txt).
        katalog_wyjsciowy: Katalog docelowy (domyślnie data/raw/).
        opoznienie: Opóźnienie między requestami w sekundach.

    Returns:
        Lista ścieżek do zapisanych plików.
    """
    if urls is None:
        urls = wczytaj_liste_url()

    if not urls:
        console.print("[yellow]Brak URL-i do pobrania.[/yellow]")
        return []

    console.print(f"[bold]Pobieranie {len(urls)} artykułów...[/bold]")
    zapisane = []

    for i, url in enumerate(urls, 1):
        try:
            console.print(f"  [{i}/{len(urls)}] {url}")
            artykul = pobierz_artykul(url)
            plik = zapisz_artykul(artykul, katalog_wyjsciowy)
            console.print(f"    ✓ Zapisano: {plik.name} ({len(artykul['markdown'])} znaków)")
            zapisane.append(plik)

            # Rate limiting (nie czekaj po ostatnim)
            if i < len(urls):
                time.sleep(opoznienie)

        except Exception as e:
            console.print(f"    [red]✗ Błąd: {e}[/red]")

    console.print(f"\n[green]Pobrano {len(zapisane)}/{len(urls)} artykułów.[/green]")
    return zapisane
