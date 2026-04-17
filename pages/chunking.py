"""Strona chunkingu i indeksowania."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("2. Chunking & Indeksowanie")
    st.markdown("Dzielenie artykułów na fragmenty i indeksowanie w Qdrant Cloud.")

    raw_dir = _root / "data" / "raw"
    existing = sorted(raw_dir.glob("*.md")) if raw_dir.exists() else []

    if not existing:
        st.warning("Brak artykułów w data/raw/. Najpierw pobierz je w zakładce Scraping.")
        return

    st.info(f"Dostępnych artykułów: {len(existing)}")

    # --- Przewodnik po metodach chunkingu ---
    with st.expander("📖 Przewodnik po wszystkich metodach chunkingu", expanded=False):
        import pandas as pd
        guide_data = [
            {
                "Metoda": "Naive (Recursive)",
                "Jak działa": "Tnie co N znaków. Próbuje dzielić po \\n\\n, \\n, '. ', ' '.",
                "Kiedy stosować": "Baseline, szybki start, proste FAQ",
                "Kiedy NIE": "Tekst ze strukturą (H2/H3), tabele",
                "Dokumenty": "Dowolne, ale bez struktury",
                "W pipeline": "✅ naive",
            },
            {
                "Metoda": "Header-based",
                "Jak działa": "Dzieli po nagłówkach H2/H3 (naturalna struktura artykułu)",
                "Kiedy stosować": "Artykuły SEO, blogi, dokumentacja z H2/H3",
                "Kiedy NIE": "Tekst bez nagłówków (plain text)",
                "Dokumenty": "Markdown, HTML z nagłówkami",
                "W pipeline": "✅ header",
            },
            {
                "Metoda": "Semantic",
                "Jak działa": "Cosine similarity między zdaniami → tnie przy zmianach tematu",
                "Kiedy stosować": "Tekst bez struktury, eseje, narracja",
                "Kiedy NIE": "Spójny tematycznie tekst (1 temat = 1 gigantyczny chunk)",
                "Dokumenty": "Dowolne, ale drogie (embedding per zdanie)",
                "W pipeline": "✅ semantic",
            },
            {
                "Metoda": "Proposition-based",
                "Jak działa": "LLM rozbija na atomowe twierdzenia (1 fakt = 1 chunk)",
                "Kiedy stosować": "Max precyzja, fact-checking, QA nad faktami",
                "Kiedy NIE": "Duży korpus (drogie — LLM per fragment), narracja",
                "Dokumenty": "Encyklopedyczne, specyfikacje, regulaminy",
                "W pipeline": "✅ proposition",
            },
            {
                "Metoda": "Parent-child (small-to-big)",
                "Jak działa": "Małe chunki do retrieval, ale do LLM trafia CAŁY parent (sekcja H2)",
                "Kiedy stosować": "Precyzyjne szukanie + pełny kontekst w odpowiedzi",
                "Kiedy NIE": "Krótkie dokumenty (parent ≈ cały dokument)",
                "Dokumenty": "Artykuły z sekcjami, dokumentacja",
                "W pipeline": "✅ parent_child",
            },
            {
                "Metoda": "Sentence-level",
                "Jak działa": "Każde zdanie = osobny chunk",
                "Kiedy stosować": "FAQ, krótkie odpowiedzi, granularne wyszukiwanie",
                "Kiedy NIE": "Długie odpowiedzi (za dużo małych chunków, gubi kontekst)",
                "Dokumenty": "FAQ, definicje, specyfikacje",
                "W pipeline": "✅ sentence",
            },
            {
                "Metoda": "Layout-aware",
                "Jak działa": "Rozpoznaje tabele, listy, nagłówki — nie tnie w środku struktury",
                "Kiedy stosować": "Markdown/HTML z tabelami, listami, blokami kodu",
                "Kiedy NIE": "Czysty tekst bez formatowania",
                "Dokumenty": "Dokumentacja techniczna, raporty z tabelami",
                "W pipeline": "✅ layout_aware",
            },
            {
                "Metoda": "Recursive (LangChain)",
                "Jak działa": "= Naive. Próbuje \\n\\n → \\n → '. ' → ' '. To samo co naive w tym pipeline.",
                "Kiedy stosować": "—",
                "Kiedy NIE": "Już masz jako 'naive'",
                "Dokumenty": "—",
                "W pipeline": "= naive",
            },
            {
                "Metoda": "Sliding window",
                "Jak działa": "Okno przesuwa się co K tokenów = naive z overlap. Inna nazwa, ta sama logika.",
                "Kiedy stosować": "—",
                "Kiedy NIE": "Już masz jako 'naive' z overlap",
                "Dokumenty": "—",
                "W pipeline": "= naive",
            },
            {
                "Metoda": "Token-based",
                "Jak działa": "Jak naive ale liczy tokeny (nie znaki). ~4 znaki/token.",
                "Kiedy stosować": "Kontrola context window LLM",
                "Kiedy NIE": "Mała różnica vs znakowy — naive wystarczy",
                "Dokumenty": "—",
                "W pipeline": "≈ naive",
            },
        ]
        df_guide = pd.DataFrame(guide_data)
        st.dataframe(df_guide, use_container_width=True, hide_index=True, height=420)

    # --- Konfiguracja chunkingu ---
    st.header("Konfiguracja")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Naive")
        naive_size = st.number_input("Rozmiar chunka (znaki)", value=1000, step=100, key="naive_size",
                                     help="Tekst jest cięty co N znaków — mechanicznie, bez analizy treści. Mniejszy rozmiar = więcej chunków, precyzyjniejszy retrieval, ale ryzyko przecięcia w środku zdania.")
        naive_overlap = st.number_input("Overlap (znaki)", value=200, step=50, key="naive_overlap",
                                        help="Ile znaków nakłada się między sąsiednimi chunkami. Zapobiega utracie kontekstu na granicy cięcia. Typowo 10-20% rozmiaru chunka (np. 100-200 przy 1000).")

    with col2:
        st.subheader("Header-based")
        header_max = st.number_input("Max rozmiar sekcji", value=2000, step=200, key="header_max",
                                     help="Dzieli tekst po nagłówkach H2/H3 — naturalna struktura artykułu. Sekcje dłuższe niż ten limit zostaną dodatkowo podzielone (jak naive, ale wewnątrz sekcji).")
        st.caption("Dzieli po H2/H3. Sekcje dłuższe niż max zostaną podzielone.")

    with col3:
        st.subheader("Semantic")
        semantic_threshold = st.slider("Percentile threshold", 50, 99, 85, key="sem_thresh",
                                       help="Chunker liczy cosine similarity między sąsiednimi zdaniami i tnie tam gdzie similarity spada. Percentile 85 = tnie tylko 15% najostrzejszych spadków (zmian tematu). Wyższy = mniej cięć = większe chunki.")
        st.caption("Wyższy = mniej chunków (łączy podobne zdania).")

    # --- Konfiguracja dodatkowych metod ---
    st.markdown("")
    col4, col5, col6, col7 = st.columns(4)

    with col4:
        st.subheader("Proposition")
        st.caption("⚠️ Claude API per fragment")
        st.caption("Brak parametrów — LLM decyduje jak rozbić na atomowe fakty.")

    with col5:
        st.subheader("Parent-Child")
        parent_child_size = st.number_input("Rozmiar child (znaki)", value=300, step=50, key="pc_size",
                                            help="Rozmiar małych chunków (children) używanych do retrieval. Do LLM trafia cały parent (sekcja H2).")
        parent_child_overlap = st.number_input("Overlap child", value=50, step=10, key="pc_overlap",
                                               help="Overlap między małymi chunkami.")

    with col6:
        st.subheader("Sentence")
        sentence_min_len = st.number_input("Min długość zdania", value=15, step=5, key="sent_min",
                                           help="Pomiń zdania krótsze niż N znaków (filtruje szum typu '...' czy 'np.').")
        st.caption("Każde zdanie = osobny chunk.")

    with col7:
        st.subheader("Layout-aware")
        layout_max = st.number_input("Max rozmiar bloku", value=2000, step=200, key="layout_max",
                                     help="Bloki (tabele, listy, sekcje) większe niż ten limit zostaną podzielone. Małe bloki łączone.")
        st.caption("Rozpoznaje tabele, listy, nagłówki.")

    # --- Wybór metod ---
    st.markdown("---")
    st.header("Indeksowanie")

    from src.chunking.registry import AVAILABLE_METHODS

    metody = st.multiselect(
        "Wybierz metody chunkingu",
        AVAILABLE_METHODS,
        default=["naive", "header", "semantic"],
        help="Każda metoda tworzy osobną kolekcję w Qdrant (np. articles_naive, articles_proposition).",
    )

    # Legenda wybranych metod
    METHOD_INFO = {
        "naive": "Tnie co N znaków z overlap. Baseline — szybki, bez analizy treści.",
        "header": "Dzieli po nagłówkach H2/H3. Idealne dla artykułów SEO ze strukturą.",
        "semantic": "Cosine similarity między zdaniami → tnie przy zmianach tematu. ⚠️ Jina API per zdanie.",
        "proposition": "LLM rozbija na atomowe fakty (1 twierdzenie = 1 chunk). ⚠️ Claude API per fragment.",
        "parent_child": "Małe chunki (300 zn) do szukania, cała sekcja H2 w metadata do generowania.",
        "sentence": "Każde zdanie = osobny chunk. Najdrobniejsza granulacja.",
        "layout_aware": "Rozpoznaje tabele, listy, nagłówki — nie tnie w środku struktury.",
    }
    if metody:
        with st.expander("ℹ️ Opis wybranych metod"):
            for m in metody:
                desc = METHOD_INFO.get(m, "")
                st.markdown(f"**{m}** — {desc}")

    # Ostrzeżenie o kosztach API
    kosztowne = [m for m in metody if m in ("proposition", "semantic")]
    if kosztowne:
        st.warning(
            f"**Uwaga na koszty API:** Metody **{', '.join(kosztowne)}** wywołują zewnętrzne API per chunk:\n"
            f"- **proposition** → Claude API (LLM rozbija tekst na atomowe fakty)\n"
            f"- **semantic** → Jina API (embedding każdego zdania do analizy similarity)\n\n"
            f"Pozostałe metody (naive, header, sentence, layout_aware, parent_child) są **darmowe** — działają czysto programatycznie, bez API."
        )

    recreate = st.checkbox("Odtwórz kolekcje od nowa (usuń istniejące)", value=False,
                           help="Usuwa istniejące kolekcje i tworzy od nowa. Użyj po zmianie artykułów lub parametrów.")

    # --- Status kolekcji ---
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()

        cols = st.columns(3)
        for i, method in enumerate(AVAILABLE_METHODS):
            name = qdrant_cfg.collection_name(method)
            with cols[i]:
                if store.collection_exists(name):
                    count = store.count_points(name)
                    st.metric(f"articles_{method}", f"{count} punktów")
                else:
                    st.metric(f"articles_{method}", "nie istnieje")
        store.close()
    except Exception as e:
        st.warning(f"Nie można połączyć z Qdrant: {e}")

    # --- Uruchom indeksowanie ---
    if st.button("🚀 Uruchom chunking i indeksowanie", type="primary", disabled=len(metody) == 0):
        from config.settings import ChunkingConfig
        from src.chunking.registry import get_chunker
        from src.chunking.base import Chunk
        from src.embeddings.jina_embed import JinaDenseEmbedder, JinaEmbeddingsLangChain
        from src.embeddings.sparse_embed import SparseBM25Embedder
        from src.vectorstore.qdrant_store import QdrantStore
        from config.settings import get_qdrant_config

        # Custom config
        config = ChunkingConfig(
            naive_chunk_size=naive_size,
            naive_chunk_overlap=naive_overlap,
            header_max_chunk_size=header_max,
            semantic_breakpoint_threshold=semantic_threshold,
        )

        # Wczytaj artykuły
        artykuly = []
        for plik in existing:
            tekst = plik.read_text(encoding="utf-8")
            if tekst.startswith("---"):
                parts = tekst.split("---", 2)
                if len(parts) >= 3:
                    tekst = parts[2].strip()
            artykuly.append({"text": tekst, "slug": plik.stem, "path": str(plik)})

        dense_embedder = JinaDenseEmbedder()
        sparse_embedder = SparseBM25Embedder()
        lc_embeddings = JinaEmbeddingsLangChain()
        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()

        all_stats = {}

        for metoda in metody:
            st.subheader(f"Metoda: {metoda.upper()}")
            progress = st.progress(0, text=f"Chunkowanie ({metoda})...")

            # Chunking
            chunker = get_chunker(
                metoda,
                config=config,
                embeddings=lc_embeddings if metoda == "semantic" else None,
            )

            all_chunks: list[Chunk] = []
            for i, art in enumerate(artykuly):
                meta = {"slug": art["slug"], "source_file": art["path"]}
                chunks = chunker.chunk(art["text"], metadata=meta)
                all_chunks.extend(chunks)
                progress.progress(
                    (i + 1) / len(artykuly) * 0.3,
                    text=f"Chunkowanie ({metoda}): {i+1}/{len(artykuly)}",
                )

            # Statystyki
            lengths = [len(c.text) for c in all_chunks]
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Chunków", len(all_chunks))
            col_b.metric("Avg długość", f"{sum(lengths)/len(lengths):.0f}" if lengths else "0")
            col_c.metric("Min", min(lengths) if lengths else 0)
            col_d.metric("Max", max(lengths) if lengths else 0)

            all_stats[metoda] = {
                "count": len(all_chunks),
                "avg": sum(lengths)/len(lengths) if lengths else 0,
                "lengths": lengths,
            }

            # Embeddingi dense
            progress.progress(0.4, text=f"Embeddingi dense ({metoda})...")
            texts = [c.text for c in all_chunks]
            dense_vectors = dense_embedder.embed_documents(texts)

            # Embeddingi sparse
            progress.progress(0.7, text=f"Embeddingi sparse ({metoda})...")
            sparse_vectors = sparse_embedder.embed_documents(texts)

            # Upsert
            progress.progress(0.9, text=f"Indeksowanie w Qdrant ({metoda})...")
            collection_name = qdrant_cfg.collection_name(metoda)
            store.create_collection(collection_name, recreate=recreate)
            store.upsert_chunks(collection_name, all_chunks, dense_vectors, sparse_vectors)

            count = store.count_points(collection_name)
            progress.progress(1.0, text=f"✓ {metoda}: {count} punktów w Qdrant")
            st.success(f"Zaindeksowano {count} chunków metodą {metoda}")

        dense_embedder.close()
        store.close()

        # Tabela podsumowująca
        if all_stats:
            st.markdown("---")
            st.subheader("Podsumowanie")
            import pandas as pd
            df = pd.DataFrame([
                {"Metoda": k, "Chunków": v["count"], "Avg długość": f"{v['avg']:.0f}"}
                for k, v in all_stats.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Box plot rozkładu długości per metoda
            import plotly.graph_objects as go
            fig_box = go.Figure()
            for metoda, data in all_stats.items():
                if data.get("lengths"):
                    fig_box.add_trace(go.Box(
                        y=data["lengths"],
                        name=metoda,
                        boxmean=True,
                    ))
            fig_box.update_layout(
                title="Rozkład długości chunków per metoda",
                yaxis_title="Długość (znaki)",
                height=400,
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Histogram nakładający wszystkie metody
            fig_hist = go.Figure()
            for metoda, data in all_stats.items():
                if data.get("lengths"):
                    fig_hist.add_trace(go.Histogram(
                        x=data["lengths"],
                        name=metoda,
                        opacity=0.6,
                        nbinsx=30,
                    ))
            fig_hist.update_layout(
                title="Histogram długości chunków (nakładający)",
                xaxis_title="Długość (znaki)",
                yaxis_title="Liczba chunków",
                barmode="overlay",
                height=400,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- Podgląd chunków ---
    st.markdown("---")
    st.header("Podgląd chunków")

    preview_article = st.selectbox(
        "Wybierz artykuł do podglądu",
        existing,
        format_func=lambda p: p.name,
        key="preview_article",
    )
    preview_method = st.radio("Metoda", AVAILABLE_METHODS, horizontal=True, key="preview_method")

    if st.button("Pokaż chunki"):
        from config.settings import ChunkingConfig
        from src.chunking.registry import get_chunker
        from src.embeddings.jina_embed import JinaEmbeddingsLangChain

        config = ChunkingConfig(
            naive_chunk_size=naive_size,
            naive_chunk_overlap=naive_overlap,
            header_max_chunk_size=header_max,
            semantic_breakpoint_threshold=semantic_threshold,
        )

        tekst = preview_article.read_text(encoding="utf-8")
        if tekst.startswith("---"):
            parts = tekst.split("---", 2)
            if len(parts) >= 3:
                tekst = parts[2].strip()

        lc_emb = JinaEmbeddingsLangChain() if preview_method == "semantic" else None
        chunker = get_chunker(preview_method, config=config, embeddings=lc_emb)
        chunks = chunker.chunk(tekst, {"slug": preview_article.stem})

        st.write(f"**{len(chunks)} chunków:**")
        for i, chunk in enumerate(chunks):
            header_info = ""
            if "h2" in chunk.metadata:
                header_info = f" | H2: {chunk.metadata['h2']}"
            if "h3" in chunk.metadata:
                header_info += f" | H3: {chunk.metadata['h3']}"

            with st.expander(f"Chunk {i+1} ({len(chunk.text)} znaków{header_info})"):
                st.text(chunk.text)
