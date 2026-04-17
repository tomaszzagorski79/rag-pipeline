# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Język

Projekt jest w języku polskim — komentarze, docstringi, nazwy zmiennych (z wyjątkiem nazw klas/modułów) i komunikaty UI po polsku.

## Uruchamianie

```bash
# Aktywacja venv (Windows)
.venv\Scripts\activate

# Streamlit UI — główny sposób użycia (22 zakładki)
streamlit run app.py --server.port 8501 --server.headless true

# Albo przez start.bat (dwuklik)
```

## CLI Pipeline (starsze skrypty, obejmują tylko Fazę 1)

```bash
python scripts/01_scrape_articles.py        # --url, --dry-run, --delay
python scripts/02_chunk_and_index.py --method all   # obsługuje tylko naive|header|semantic
python scripts/03_query.py                  # interaktywne zapytania
python scripts/04_run_evaluation.py         # RAGAS
python scripts/05_compare_methods.py
```

Wszystkie nowsze funkcje (PageIndex, Graph RAG, Agentic, Context Engineering, itd.) są dostępne **tylko z UI** — skrypty CLI nie były rozszerzane razem z pipeline. Preferuj Streamlit.

Skrypty wymagają `sys.path.insert(0, root)` — uruchamiaj z katalogu głównego projektu.

## Architektura

Pipeline RAG z **22 zakładkami** edukacyjnymi pokazującymi kolejne architektury RAG od Naive do Context Engineering.

### Stos technologiczny

| Warstwa | Tech |
|---------|------|
| Vector DB | Qdrant Cloud (dual vectors: dense + sparse + IDF) |
| Graph DB | Neo4j Aura (opcjonalnie — dla Graph RAG, Hybrid V+G, Context Engineering) |
| Dense embeddings | Jina v5 text-small (API, 1024d) |
| Sparse embeddings | FastEmbed BM25 (`Qdrant/bm25`) |
| LLM | Claude Sonnet 4 (odpowiedzi) + Claude Haiku 4 (drafter w Speculative RAG) |
| Re-ranking | FlashRank (lokalne, cross-encoder) |
| Benchmarki | Gemini, multilingual-e5-large, BGE-m3 (opcjonalnie) |
| UI | Streamlit + Plotly + matplotlib (Venn) |

### Główny przepływ danych

```
URL/sitemap → Jina Reader → markdown (data/raw/)
  → Chunker (7 metod) → Chunk[]
    → Jina v5 dense + FastEmbed sparse
      → Qdrant (articles_{method}) — dual vectors
        → HybridRetriever (prefetch dense+sparse → RRF) → SearchResult[]
          → [opcjonalnie: re-rank, HyDE, CRAG, FLARE, agent...]
            → ClaudeGenerator → odpowiedź
              → RAGAS / Hallucination / CRQ → metryki
```

### Architektury RAG dostępne w pipeline

Zakładki Streamlit mapują się 1:1 na moduły w `src/`:

| Zakładka | Moduł | Technika |
|----------|-------|----------|
| 4 Zapytania | `src/retrieval/` | Hybrid search (BM25+vector+RRF) |
| 5 Re-ranking | `src/reranking/` | FlashRank cross-encoder |
| 6 HyDE | `src/hyde/` | Hypothetical Document Embeddings |
| 7 Adaptive RAG | `src/adaptive/` | Query classifier + tier routing |
| 8 CRAG | `src/crag/` | Corrective RAG + knowledge strip |
| 9 Ewaluacja | `src/evaluation/` | RAGAS z Claude jako LLM evaluator |
| 10 Hallucination | `src/hallucination/` | Claim-by-claim verification |
| 11 Benchmarki | `src/benchmarks/` | hit@k + MRR na wielu modelach embed |
| 12 CRQ | `src/crq/` | Content Retrieval Quality (density/BLUF/chunking/EAV) |
| 13 PageIndex | `src/pageindex/` | Vectorless reasoning-based RAG |
| 14 Agentic RAG | `src/agentic/` | Claude tool_use + ReAct loop |
| 15 Graph RAG | `src/graph_rag/` | Neo4j + EAV triples extraction |
| 16 Hybrid V+G | `pages/hybrid_vg.py` | Vector + Graph równolegle |
| 17 Context Engineering | `src/context_eng/` | 6 komponentów: orchestrator, augmenter, memory, facets |
| 18 RAG-Fusion | `src/rag_fusion/` | Multi-query + RRF |
| 19 FLARE | `src/flare/` | Forward-looking active retrieval |
| 20 RAPTOR | `src/raptor/` | Hierarchiczny indeks drzewiasty |
| 21 Speculative RAG | `src/speculative/` | Drafter (Haiku) + Verifier (Sonnet) |

### Kluczowe abstrakcje

- **`ChunkerBase` ABC + `Chunk` dataclass** (`src/chunking/base.py`) — kontrakt danych w całym pipeline
- **`SearchResult` dataclass** (`src/retrieval/hybrid_search.py`) — `text`, `score`, `metadata`, `chunk_id`. Używany przez wszystkie retrievery i konsumowany przez każdą technikę RAG
- **`JinaEmbeddingsLangChain` adapter** (`src/embeddings/jina_embed.py`) — implementuje interfejs LangChain `Embeddings`; używany przez SemanticChunker i RAGAS
- **Registry pattern** (`src/chunking/registry.py`) — `get_chunker(name)` + `AVAILABLE_METHODS` (pojedyncze źródło prawdy, używane przez 16 plików)
- **Deterministic UUID5** w `qdrant_store.py` — re-indeksowanie nadpisuje zamiast duplikować

### Dual vectors w Qdrant

Każdy punkt ma wektor `dense` (Jina, cosine) + `sparse` (BM25, IDF modifier w Qdrant). Hybrid search robi prefetch z obu i łączy przez Reciprocal Rank Fusion.

### Osobne kolekcje per metoda chunkingu

7 metod → 7 kolekcji `articles_{method}` (naive, header, semantic, proposition, parent_child, sentence, layout_aware). Tytuły mają osobną kolekcję `articles_titles` (dense only, bez sparse).

### Zasada: `AVAILABLE_METHODS` jako single source of truth

Lista 7 metod chunkingu jest w `src/chunking/registry.py`. **Nigdy nie hardcodować** `["naive", "header", "semantic"]` w pages/ — zawsze `from src.chunking.registry import AVAILABLE_METHODS` (15 stron już tak robi).

## Konfiguracja

Centralna w `config/settings.py` — dataclassy ładowane z `.env` (`override=True`).

**Wymagane:** `QDRANT_URL`, `QDRANT_API_KEY`, `JINA_API_KEY`, `ANTHROPIC_API_KEY`

**Opcjonalne:** `GOOGLE_API_KEY` (benchmarki Gemini), `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` (Graph RAG, Hybrid V+G, Context Engineering)

## Streamlit UI

`app.py` — routing sidebar radio do `pages/{name}.py`. Każda strona ma funkcję `render()`. Strony używają lazy imports (wewnątrz `if st.button`) żeby nie ładować ciężkich zależności przy starcie.

`.streamlit/config.toml` — wyłącza domyślną wielostronicową nawigację Streamlit (`showSidebarNavigation = false`) i telemetrię.

Każda strona RAG zawiera:
1. Expander "Czym jest X?" z sekcjami **Kiedy używać / Kiedy NIE**
2. Tooltips (`help="..."`) przy każdym widgetcie
3. Sprawdzenie dostępności kolekcji przez `AVAILABLE_METHODS`
4. Wizualizacje (Plotly: scatter/sankey/heatmap/treemap/timeline + UMAP + Venn)

## Dane

- `data/raw/*.md` — artykuły z YAML frontmatter (`title`, `source_url`, `scraped_at`), treść po `---`
- `data/article_urls.txt` — lista URL-i (jeden na linię, `#` = komentarz)
- `data/test_set.json` — pytania testowe: `[{question, ground_truth, source_article}]`
- `data/results/evaluation_*.json` — wyniki RAGAS
- `data/results/benchmark_*.json` — wyniki benchmarków embeddingów
- `data/results/crq_*.json` — wyniki CRQ per artykuł
- `data/memory/session_*.json` — episodic memory Context Engineering

Wszystko poza `article_urls.txt` i `test_set.json` jest w `.gitignore`.

## Scraper

Jina Reader (`r.jina.ai`) + post-processing: `_wyczysc_nawigacje()` wycina menu/footer IdoSell szukając pierwszego H2 i footer markers. Rate limit 3s między requestami. UI wspiera też import z sitemap.xml (max 300 URL-i, URL lub upload pliku).

## Dodawanie nowej metody chunkingu

1. Utwórz klasę dziedziczącą z `ChunkerBase` w `src/chunking/`
2. Zaimplementuj `name` property i `chunk()` method
3. Dodaj do `registry.py` w `get_chunker()` i `AVAILABLE_METHODS`
4. Kolekcja Qdrant utworzy się automatycznie jako `articles_{name}`
5. Wszystkie 15 stron RAG **automatycznie** zobaczą nową kolekcję w selectboxach (dzięki `AVAILABLE_METHODS`)

## Dodawanie nowej techniki RAG

1. Utwórz moduł w `src/{nazwa}/` z logiką
2. Utwórz stronę w `pages/{nazwa}.py` z funkcją `render()`
3. Dodaj pozycję do `strona = st.sidebar.radio(...)` w `app.py`
4. Dodaj `elif` blok do routingu w `app.py`
5. Strona MUSI zawierać: expander "Czym jest X? / Kiedy używać / Kiedy NIE", tooltips przy widgetach, import `AVAILABLE_METHODS` do selectboxa kolekcji

## Uwagi o kosztach API

- **Darmowe chunkery:** naive, header, sentence, layout_aware, parent_child (czysto programatyczne)
- **Płatne chunkery:** `semantic` (Jina API per zdanie), `proposition` (Claude API per fragment)
- **Tytuły** (zakładka 2) zużywają Jina tylko dla wybranych tytułów — tanie
- **Agentic RAG, FLARE, CRQ, Hallucination** robią wiele wywołań Claude — drogie per zapytanie
- `ewaluacja.py` używa Claude jako judge w RAGAS — koszt rośnie z liczbą pytań
