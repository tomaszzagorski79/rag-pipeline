"""Strona Benchmarki embeddingów — porównanie modeli na polskich treściach."""

import json
import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("11. Benchmarki embeddingów")

    with st.expander("Czym jest benchmark embeddingów?", expanded=False):
        st.markdown("""
**Cel:** Porównać jak różne modele embeddingowe radzą sobie z **polskimi** treściami e-commerce.

**Metryki:**
- **Hit@K** — czy prawidłowy artykuł jest w top-K wynikach? (K=1,3,5,10)
- **MRR** (Mean Reciprocal Rank) — na której pozycji pojawia się pierwszy trafiony chunk?

**Modele:**
- Jina v5 text-small (API, 1024d)
- multilingual-e5-large (lokalne, 1024d)
- BGE-M3 (lokalne, 1024d)
- OpenAI text-embedding-3-small (API, 1536d)

**Uwaga:** Lokalne modele (e5, bge-m3) wymagają `sentence-transformers` i ~3GB RAM.
        """)

    raw_dir = _root / "data" / "raw"
    results_dir = _root / "data" / "results"
    test_set_file = _root / "data" / "test_set.json"

    # Sprawdź dane
    if not test_set_file.exists():
        st.warning("Brak pytań testowych (data/test_set.json). Dodaj je w zakładce Ewaluacja RAGAS.")
        return

    test_data = json.loads(test_set_file.read_text(encoding="utf-8"))
    if len(test_data) < 2:
        st.warning("Dodaj więcej pytań testowych (min. 2).")
        return

    # Wczytaj chunki (header method jako baseline)
    artykuly = []
    if raw_dir.exists():
        for plik in sorted(raw_dir.glob("*.md")):
            tekst = plik.read_text(encoding="utf-8")
            if tekst.startswith("---"):
                parts = tekst.split("---", 2)
                if len(parts) >= 3:
                    tekst = parts[2].strip()
            artykuly.append({"text": tekst, "slug": plik.stem})

    if not artykuly:
        st.warning("Brak artykułów. Pobierz je w zakładce Scraping.")
        return

    # Chunk artykuły (header method)
    from src.chunking.header_based import HeaderChunker
    chunker = HeaderChunker()
    chunks = []
    for art in artykuly:
        art_chunks = chunker.chunk(art["text"], {"slug": art["slug"]})
        for c in art_chunks:
            chunks.append({"text": c.text, "slug": art["slug"]})

    st.info(f"Pytań testowych: {len(test_data)} | Chunków: {len(chunks)} | Artykułów: {len(artykuly)}")

    # Wybór modeli
    st.subheader("Wybierz modele")

    from src.benchmarks.embedding_benchmark import AVAILABLE_MODELS
    import os

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_st = True
    try:
        import sentence_transformers
    except ImportError:
        has_st = False

    selected_models = []
    col1, col2 = st.columns(2)

    with col1:
        if st.checkbox("Jina v5 (API)", value=True, key="bm_jina"):
            selected_models.append("jina-v5")
        if st.checkbox("OpenAI (API)", value=has_openai, disabled=not has_openai, key="bm_openai"):
            selected_models.append("openai")
            if not has_openai:
                st.caption("Wymaga OPENAI_API_KEY w .env")

    with col2:
        if st.checkbox("multilingual-e5-large (lokalne)", value=has_st, disabled=not has_st, key="bm_e5"):
            selected_models.append("e5-large")
        if st.checkbox("BGE-M3 (lokalne)", value=False, disabled=not has_st, key="bm_bge"):
            selected_models.append("bge-m3")
        if not has_st:
            st.caption("Wymaga: pip install sentence-transformers")

    if st.button("🏁 Uruchom benchmark", type="primary", disabled=len(selected_models) == 0):
        from src.benchmarks.embedding_benchmark import run_benchmark

        progress = st.progress(0)

        def update_progress(p, text):
            progress.progress(p, text=text)

        with st.spinner("Benchmark w toku..."):
            results = run_benchmark(
                model_keys=selected_models,
                test_questions=test_data,
                chunks=chunks,
                progress_callback=update_progress,
            )

        # Tabela wyników
        st.markdown("---")
        st.subheader("Wyniki")

        import pandas as pd

        rows = []
        for r in results:
            rows.append({
                "Model": r.model_name,
                "Hit@1": f"{r.hit_at_1:.2%}",
                "Hit@3": f"{r.hit_at_3:.2%}",
                "Hit@5": f"{r.hit_at_5:.2%}",
                "Hit@10": f"{r.hit_at_10:.2%}",
                "MRR": f"{r.mrr:.3f}",
                "Dim": r.embedding_dim,
                "ms/chunk": r.avg_embed_time_ms,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Wykres
        import plotly.graph_objects as go

        fig = go.Figure()
        for r in results:
            fig.add_trace(go.Bar(
                name=r.model_name,
                x=["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"],
                y=[r.hit_at_1, r.hit_at_3, r.hit_at_5, r.hit_at_10, r.mrr],
            ))
        fig.update_layout(barmode="group", title="Porównanie modeli", height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Zapis
        results_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = results_dir / f"benchmark_{ts}.json"
        out_file.write_text(
            json.dumps([{
                "model": r.model_name,
                "hit_at_1": r.hit_at_1,
                "hit_at_3": r.hit_at_3,
                "hit_at_5": r.hit_at_5,
                "hit_at_10": r.hit_at_10,
                "mrr": r.mrr,
                "dim": r.embedding_dim,
                "ms_per_chunk": r.avg_embed_time_ms,
            } for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        st.caption(f"Zapisano: {out_file.name}")

    # Pokaż poprzednie wyniki
    st.markdown("---")
    prev_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True) if results_dir.exists() else []
    if prev_files:
        st.subheader("Poprzednie benchmarki")
        for f in prev_files[:5]:
            with st.expander(f.name):
                data = json.loads(f.read_text(encoding="utf-8"))
                import pandas as pd
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
