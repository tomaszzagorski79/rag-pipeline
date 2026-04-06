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

    # --- Konfiguracja chunkingu ---
    st.header("Konfiguracja")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Naive")
        naive_size = st.number_input("Rozmiar chunka (znaki)", value=1000, step=100, key="naive_size")
        naive_overlap = st.number_input("Overlap (znaki)", value=200, step=50, key="naive_overlap")

    with col2:
        st.subheader("Header-based")
        header_max = st.number_input("Max rozmiar sekcji", value=2000, step=200, key="header_max")
        st.caption("Dzieli po H2/H3. Sekcje dłuższe niż max zostaną podzielone.")

    with col3:
        st.subheader("Semantic")
        semantic_threshold = st.slider("Percentile threshold", 50, 99, 85, key="sem_thresh")
        st.caption("Wyższy = mniej chunków (łączy podobne zdania).")

    # --- Wybór metod ---
    st.markdown("---")
    st.header("Indeksowanie")

    metody = st.multiselect(
        "Wybierz metody chunkingu",
        ["naive", "header", "semantic"],
        default=["naive", "header", "semantic"],
    )

    recreate = st.checkbox("Odtwórz kolekcje od nowa (usuń istniejące)", value=False)

    # --- Status kolekcji ---
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()

        cols = st.columns(3)
        for i, method in enumerate(["naive", "header", "semantic"]):
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

            all_stats[metoda] = {"count": len(all_chunks), "avg": sum(lengths)/len(lengths) if lengths else 0}

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

    # --- Podgląd chunków ---
    st.markdown("---")
    st.header("Podgląd chunków")

    preview_article = st.selectbox(
        "Wybierz artykuł do podglądu",
        existing,
        format_func=lambda p: p.name,
        key="preview_article",
    )
    preview_method = st.radio("Metoda", ["naive", "header", "semantic"], horizontal=True, key="preview_method")

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
