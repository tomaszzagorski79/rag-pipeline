# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Język

Projekt jest w języku polskim — komentarze, docstringi, nazwy zmiennych (z wyjątkiem nazw klas/modułów) i komunikaty UI po polsku.

## Uruchamianie

```bash
# Aktywacja venv
.venv\Scripts\activate

# Streamlit UI (główny sposób użycia)
streamlit run app.py --server.port 8501 --server.headless true

# Lub przez start.bat (Windows, dwuklik)
```

## CLI Pipeline

```bash
python scripts/01_scrape_articles.py                    # scraping (--url, --dry-run, --delay)
python scripts/02_chunk_and_index.py --method all       # chunking + indeksowanie (--recreate)
python scripts/03_query.py                              # interaktywne zapytania (--method, --limit)
python scripts/04_run_evaluation.py --method all        # ewaluacja RAGAS (--limit)
python scripts/05_compare_methods.py                    # porównanie wyników (--results-file)
```

Skrypty wymagają `sys.path.insert(0, root)` — uruchamiaj z katalogu głównego projektu.

## Architektura

Pipeline RAG z 3 wymiennymi metodami chunkingu, hybrid search (dense+sparse → RRF) i ewaluacją RAGAS.

**Przepływ danych:**
```
URL → Jina Reader → markdown (data/raw/)
  → Chunker (naive|header|semantic) → Chunk[]
    → Jina v5 dense (1024d) + FastEmbed BM25 sparse
      → Qdrant Cloud (3 kolekcje: articles_{method})
        → HybridRetriever (prefetch+RRF) → konteksty
          → ClaudeGenerator → odpowiedź
            → RAGAS evaluate → scores
```

**Kluczowe abstrakcje:**
- `ChunkerBase` ABC + `Chunk` dataclass (`src/chunking/base.py`) — kontrakt danych w całym pipeline
- `JinaEmbeddingsLangChain` adapter (`src/embeddings/jina_embed.py`) — używany przez SemanticChunker i RAGAS
- Registry pattern w `src/chunking/registry.py` — `get_chunker(name)` zwraca skonfigurowaną instancję
- Deterministic UUID5 w `qdrant_store.py` — re-indeksowanie nadpisuje zamiast duplikować

**Dual vectors w Qdrant:** każdy punkt ma wektor `dense` (Jina, cosine) + `sparse` (BM25, IDF modifier). Hybrid search robi prefetch z obu i łączy przez Reciprocal Rank Fusion.

**3 osobne kolekcje** (nie tagi) per metoda chunkingu — izolacja porównań.

## Konfiguracja

Centralna w `config/settings.py` — dataclassy ładowane z `.env` (override=True). Wymagane zmienne:
`QDRANT_URL`, `QDRANT_API_KEY`, `JINA_API_KEY`, `ANTHROPIC_API_KEY`

## Streamlit UI

`app.py` → routing do `pages/` (przeglad, scraping, chunking, zapytania, ewaluacja). Każda strona ma funkcję `render()`. Sidebar: nawigacja + status kluczy API.

## Dane

- `data/raw/*.md` — artykuły z YAML frontmatter (title, source_url, scraped_at), treść po `---`
- `data/test_set.json` — pytania testowe: `[{question, ground_truth, source_article}]`
- `data/results/evaluation_*.json` — wyniki RAGAS per metoda

## Scraper

Jina Reader (`r.jina.ai`) + post-processing: `_wyczysc_nawigacje()` wycina menu/footer IdoSell szukając pierwszego H2 i footer markers. Rate limit 3s między requestami.

## Dodawanie nowej metody chunkingu

1. Utwórz klasę dziedziczącą z `ChunkerBase` w `src/chunking/`
2. Zaimplementuj `name` property i `chunk()` method
3. Dodaj do `registry.py` w `get_chunker()` i `AVAILABLE_METHODS`
4. Kolekcja Qdrant utworzy się automatycznie jako `articles_{name}`
