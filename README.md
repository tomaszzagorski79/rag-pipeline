# RAG Pipeline — Kompletny Przewodnik Architektur

Edukacyjny pipeline RAG (Retrieval-Augmented Generation) z **22 zakładkami Streamlit** pokazującymi kolejne architektury od Naive RAG do Context Engineering. Każda technika ma interaktywne demo, wizualizacje i sekcję "kiedy używać / kiedy nie".

Projekt powstał jako lokalne laboratorium do testowania technik RAG na polskich artykułach e-commerce (blog IdoSell).

## Co zawiera

### Fundament
- **Hybrid Search** — BM25 (sparse) + Jina Embeddings v5 (dense) → Reciprocal Rank Fusion
- **7 metod chunkingu** — naive, header-based, semantic, proposition-based, parent-child, sentence-level, layout-aware
- **RAGAS** — faithfulness, answer relevancy, context precision

### Zaawansowane architektury (zakładki 5-22)
| # | Technika | Co pokazuje |
|---|----------|-------------|
| 5 | Re-ranking | FlashRank cross-encoder po hybrid search |
| 6 | HyDE | Hypothetical Document Embeddings |
| 7 | Adaptive RAG | Routing: no_retrieval / simple / medium / complex |
| 8 | CRAG | Corrective RAG z knowledge strip |
| 10 | Hallucination Detection | Claim-by-claim weryfikacja |
| 11 | Benchmarki | Jina v5 vs Gemini vs multilingual-e5 vs BGE-m3 |
| 12 | CRQ Scoring | Content Retrieval Quality (density/BLUF/chunking/EAV) |
| 13 | PageIndex | Vectorless reasoning-based RAG |
| 14 | Agentic RAG | Claude tool_use + ReAct loop |
| 15 | Graph RAG | Neo4j + EAV triples |
| 16 | Hybrid Vector+Graph | Qdrant + Neo4j równolegle |
| 17 | Context Engineering | 6 komponentów: orchestrator, augmenter, memory, facets, budget |
| 18 | RAG-Fusion | Multi-query + RRF |
| 19 | FLARE | Forward-looking active retrieval |
| 20 | RAPTOR | Hierarchiczny indeks drzewiasty |
| 21 | Speculative RAG | Drafter (Haiku) + Verifier (Sonnet) |
| 22 | Decision Framework | Tabela porównawcza "jaki RAG kiedy" |

### Wizualizacje
14 interaktywnych wykresów (Plotly): UMAP 2D scatter, Sankey decision flow, treemap RAPTOR, heatmap RAG-Fusion, Venn vector×graph, scatter faithfulness×relevancy, line chart pewności FLARE i inne.

## Stos technologiczny

| Warstwa | Tech |
|---------|------|
| Vector DB | Qdrant Cloud (dual vectors: dense + sparse + IDF) |
| Graph DB | Neo4j Aura (opcjonalne) |
| Dense embeddings | Jina v5 text-small (API, 1024d) |
| Sparse embeddings | FastEmbed BM25 |
| LLM | Claude Sonnet 4 + Haiku 4 |
| Re-ranking | FlashRank (lokalne, cross-encoder) |
| UI | Streamlit + Plotly |
| Python | 3.12 |

## Wymagane klucze API

**Must have:**
- `QDRANT_URL`, `QDRANT_API_KEY` — [Qdrant Cloud](https://cloud.qdrant.io/) (free tier: 1GB RAM)
- `JINA_API_KEY` — [jina.ai](https://jina.ai/) (free tier: 10M tokenów)
- `ANTHROPIC_API_KEY` — [anthropic.com](https://console.anthropic.com/)

**Opcjonalne:**
- `GOOGLE_API_KEY` — [Google AI Studio](https://aistudio.google.com/apikey) (tylko benchmarki Gemini)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — [Neo4j Aura](https://console.neo4j.io/) (tylko Graph RAG, Hybrid V+G, Context Engineering)

## Instalacja

### Windows
```bash
git clone https://github.com/tomaszzagorski79/rag-pipeline.git
cd rag-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Uzupełnij .env kluczami API
```

### Mac / Linux
```bash
git clone https://github.com/tomaszzagorski79/rag-pipeline.git
cd rag-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Uzupełnij .env kluczami API
```

## Uruchomienie

```bash
# Windows
start.bat

# Mac / Linux
./start.sh

# Lub bezpośrednio
streamlit run app.py
```

Otwórz przeglądarkę na `http://localhost:8501`.

## Workflow pipeline

1. **Scraping** — wklej URL-e (lub sitemap.xml) w zakładce "1. Scraping" → pobierz artykuły do `data/raw/`
2. **Embeddingi tytułów** (opcjonalne) — UMAP/t-SNE 2D wizualizacja
3. **Chunking & Indeksowanie** — wybierz metody (1-7) → automatyczne tworzenie kolekcji Qdrant `articles_{method}`
4. **Zapytania** — hybrid search z filtrem score
5-22. **Testuj kolejne techniki RAG** — każda zakładka ma pełne demo z wizualizacją

## Uwagi o kosztach API

**Darmowe chunkery** (czysto programatyczne): `naive`, `header`, `sentence`, `layout_aware`, `parent_child`

**Płatne chunkery**:
- `semantic` — Jina API per zdanie (Matryoshka embeddings)
- `proposition` — Claude API per fragment

**Drogie techniki RAG** (wiele wywołań Claude):
- Agentic RAG (5-20 wywołań per pytanie)
- FLARE (wywołanie per zdanie)
- CRQ Scoring (4 wywołania per artykuł)
- Hallucination Detection (1 + N per odpowiedź)
- RAGAS evaluation (koszt rośnie z liczbą pytań)

## Struktura projektu

```
rag-pipeline/
├── app.py                    # Streamlit router (22 zakładki)
├── config/settings.py        # Centralna konfiguracja (.env → dataclasses)
├── pages/                    # Strony Streamlit (1:1 z zakładkami)
├── src/
│   ├── scraper/              # Jina Reader + parsing
│   ├── chunking/             # 7 metod + registry (single source of truth)
│   ├── embeddings/           # Jina dense + FastEmbed sparse
│   ├── vectorstore/          # Qdrant (dual vectors) + tytuły
│   ├── retrieval/            # HybridRetriever (BM25+vector+RRF)
│   ├── generation/           # Claude API wrapper
│   ├── evaluation/           # RAGAS integration
│   ├── reranking/            # FlashRank cross-encoder
│   ├── hyde/                 # Hypothetical Document Embeddings
│   ├── adaptive/             # Query classifier + routing
│   ├── crag/                 # Corrective RAG
│   ├── hallucination/        # Claim-by-claim verification
│   ├── benchmarks/           # Porównanie modeli embeddingowych
│   ├── crq/                  # Content Retrieval Quality
│   ├── pageindex/            # Vectorless tree reasoning
│   ├── agentic/              # Claude tool_use + ReAct
│   ├── graph_rag/            # Neo4j + EAV extraction
│   ├── context_eng/          # Orchestrator + memory + facets
│   ├── rag_fusion/           # Multi-query + RRF
│   ├── flare/                # Active retrieval
│   ├── raptor/               # Hierarchical tree index
│   └── speculative/          # Drafter + verifier
├── scripts/                  # Starsze CLI (tylko Faza 1)
├── data/
│   ├── article_urls.txt      # Lista URL-i do scrapowania
│   ├── test_set.json         # Pytania testowe do RAGAS
│   ├── raw/                  # [gitignored] Scrapowane artykuły
│   ├── results/              # [gitignored] Wyniki ewaluacji
│   └── memory/               # [gitignored] Episodic memory
└── .streamlit/config.toml    # Wyłączenie default multipage nav + telemetria
```

## Edukacyjny kontekst

Pipeline powstał jako towarzysz roadmapy **"Od Senior SEO Specialist do AI Search & Retrieval Consultant"** — praktyczne laboratorium do testowania hipotez o RAG w kontekście polskiego e-commerce i AI Search Optimization (GEO).

Każda zakładka odpowiada konkretnej architekturze RAG z literatury (Lewis 2020, HyDE 2022, ReAct 2023, Self-RAG 2023, CRAG 2024, GraphRAG 2024, RAPTOR 2024, Speculative RAG 2024, Context Engineering 2025).

## Licencja

MIT License — patrz [LICENSE](LICENSE).

## Autor

Tomasz Zagórski — Senior SEO & AI Search Engineer.
GitHub: [@tomaszzagorski79](https://github.com/tomaszzagorski79)

## Wkład

Pull requesty mile widziane. Przed PR:
1. Przeczytaj `CLAUDE.md` (architektura)
2. Nowe techniki RAG → osobna zakładka + sekcja "kiedy używać / kiedy nie"
3. Nowe chunkery → dodaj do `src/chunking/registry.py` (`AVAILABLE_METHODS`)
