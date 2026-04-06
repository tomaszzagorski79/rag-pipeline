"""Skrypt do chunkingu artykułów i indeksowania w Qdrant.

Użycie:
    python scripts/02_chunk_and_index.py --method all              # wszystkie metody
    python scripts/02_chunk_and_index.py --method naive             # tylko naiwny
    python scripts/02_chunk_and_index.py --method header semantic   # wybrane metody
    python scripts/02_chunk_and_index.py --method all --recreate    # wyczyść i zaindeksuj od nowa
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from config.settings import get_chunking_config, get_qdrant_config
from src.chunking.base import Chunk
from src.chunking.registry import AVAILABLE_METHODS, get_chunker
from src.embeddings.jina_embed import JinaDenseEmbedder, JinaEmbeddingsLangChain
from src.embeddings.sparse_embed import SparseBM25Embedder
from src.vectorstore.qdrant_store import QdrantStore

console = Console()


def wczytaj_artykuly(katalog: Path) -> list[dict]:
    """Wczytaj artykuły markdown z katalogu data/raw/.

    Returns:
        Lista dict z kluczami 'text', 'slug', 'path'.
    """
    artykuly = []
    for plik in sorted(katalog.glob("*.md")):
        tekst = plik.read_text(encoding="utf-8")

        # Oddziel frontmatter od treści
        if tekst.startswith("---"):
            parts = tekst.split("---", 2)
            if len(parts) >= 3:
                tekst = parts[2].strip()

        slug = plik.stem
        artykuly.append({"text": tekst, "slug": slug, "path": str(plik)})

    return artykuly


def chunk_and_index(
    metody: list[str],
    recreate: bool = False,
) -> dict[str, list[Chunk]]:
    """Główna logika: chunk → embed → upsert.

    Args:
        metody: Lista metod chunkingu do wykonania.
        recreate: Czy odtwarzać kolekcje od nowa.

    Returns:
        Słownik {metoda: lista_chunków} ze statystykami.
    """
    from config.settings import get_paths

    paths = get_paths()
    qdrant_cfg = get_qdrant_config()

    # Wczytaj artykuły
    artykuly = wczytaj_artykuly(paths.raw_articles)
    if not artykuly:
        console.print("[red]Brak artykułów w data/raw/. Uruchom najpierw 01_scrape_articles.py[/red]")
        return {}

    console.print(f"[bold]Załadowano {len(artykuly)} artykułów[/bold]\n")

    # Inicjalizacja embeddingów
    dense_embedder = JinaDenseEmbedder()
    sparse_embedder = SparseBM25Embedder()
    lc_embeddings = JinaEmbeddingsLangChain()  # dla SemanticChunker
    store = QdrantStore()

    all_stats = {}

    for metoda in metody:
        console.print(f"\n[bold cyan]═══ Metoda: {metoda.upper()} ═══[/bold cyan]")

        # 1. Utwórz chunker
        chunker = get_chunker(
            metoda,
            embeddings=lc_embeddings if metoda == "semantic" else None,
        )

        # 2. Chunk wszystkich artykułów
        all_chunks: list[Chunk] = []
        for art in tqdm(artykuly, desc=f"Chunking ({metoda})"):
            meta = {"slug": art["slug"], "source_file": art["path"]}
            chunks = chunker.chunk(art["text"], metadata=meta)
            all_chunks.extend(chunks)

        console.print(f"  Chunków: {len(all_chunks)}")
        if all_chunks:
            lengths = [len(c) for c in all_chunks]
            console.print(
                f"  Długość: avg={sum(lengths)/len(lengths):.0f}, "
                f"min={min(lengths)}, max={max(lengths)}"
            )

        # 3. Embed (dense + sparse)
        console.print("  Generowanie embeddingów dense (Jina)...")
        texts = [c.text for c in all_chunks]
        dense_vectors = dense_embedder.embed_documents(texts)

        console.print("  Generowanie embeddingów sparse (BM25)...")
        sparse_vectors = sparse_embedder.embed_documents(texts)

        # 4. Upsert do Qdrant
        collection_name = qdrant_cfg.collection_name(metoda)
        store.create_collection(collection_name, recreate=recreate)
        store.upsert_chunks(collection_name, all_chunks, dense_vectors, sparse_vectors)

        # Statystyki
        count = store.count_points(collection_name)
        console.print(f"  ✓ Kolekcja '{collection_name}': {count} punktów")

        all_stats[metoda] = all_chunks

    # Tabela podsumowująca
    _wyswietl_podsumowanie(all_stats)

    store.close()
    dense_embedder.close()
    return all_stats


def _wyswietl_podsumowanie(stats: dict[str, list[Chunk]]) -> None:
    """Wyświetl tabelę porównawczą metod chunkingu."""
    table = Table(title="\nPodsumowanie chunkingu")
    table.add_column("Metoda", style="cyan")
    table.add_column("Chunków", justify="right")
    table.add_column("Avg długość", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for metoda, chunks in stats.items():
        if chunks:
            lengths = [len(c) for c in chunks]
            table.add_row(
                metoda,
                str(len(chunks)),
                f"{sum(lengths)/len(lengths):.0f}",
                str(min(lengths)),
                str(max(lengths)),
            )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Chunking i indeksowanie artykułów")
    parser.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        help="Metody chunkingu: naive, header, semantic, all",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Usuń istniejące kolekcje i zaindeksuj od nowa",
    )
    args = parser.parse_args()

    metody = AVAILABLE_METHODS if "all" in args.method else args.method

    # Walidacja
    for m in metody:
        if m not in AVAILABLE_METHODS:
            console.print(f"[red]Nieznana metoda: {m}. Dostępne: {AVAILABLE_METHODS}[/red]")
            return

    chunk_and_index(metody, recreate=args.recreate)


if __name__ == "__main__":
    main()
