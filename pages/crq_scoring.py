"""Strona CRQ Scoring — ocena jakości artykułów pod kątem retrieval."""

import json
import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def _wczytaj_artykuly(raw_dir: Path) -> list[dict]:
    """Wczytaj artykuły z data/raw/."""
    artykuly = []
    if not raw_dir.exists():
        return artykuly
    for plik in sorted(raw_dir.glob("*.md")):
        tekst = plik.read_text(encoding="utf-8")
        title = plik.stem
        content = tekst
        if tekst.startswith("---"):
            parts = tekst.split("---", 2)
            if len(parts) >= 3:
                import yaml
                try:
                    meta = yaml.safe_load(parts[1])
                    title = meta.get("title", plik.stem)
                except Exception:
                    pass
                content = parts[2].strip()
        artykuly.append({
            "title": title,
            "slug": plik.stem,
            "content": content,
            "path": str(plik),
        })
    return artykuly


def render():
    st.title("12. CRQ Scoring (Content Retrieval Quality)")

    with st.expander("Czym jest CRQ?", expanded=False):
        st.markdown("""
**CRQ** ocenia jakość artykułu pod kątem tego, jak dobrze będzie działał w pipeline RAG:

- **Information Density (30%)** — stosunek faktów do ogólników
- **BLUF compliance (25%)** — czy sekcje H2 zaczynają się od kluczowej informacji
- **Chunking Quality (20%)** — czy struktura artykułu daje dobre chunki
- **EAV coverage (25%)** — gęstość trójek Entity-Attribute-Value (fakty)

**Wynik 0-100** per wymiar + overall CRQ + rekomendacje.
        """)

    raw_dir = _root / "data" / "raw"
    results_dir = _root / "data" / "results"
    artykuly = _wczytaj_artykuly(raw_dir)

    if not artykuly:
        st.warning("Brak artykułów. Pobierz je w zakładce Scraping.")
        return

    tab_single, tab_dashboard = st.tabs(["Analiza artykułu", "Dashboard"])

    with tab_single:
        _render_single(artykuly, results_dir)

    with tab_dashboard:
        _render_dashboard(artykuly, results_dir)


def _render_single(artykuly, results_dir):
    """Analiza pojedynczego artykułu."""

    selected = st.selectbox(
        "Wybierz artykuł",
        artykuly,
        format_func=lambda a: a["title"],
        key="crq_article",
    )

    if st.button("📊 Analizuj CRQ", type="primary"):
        from src.crq.crq_scorer import CRQScorer

        scorer = CRQScorer()

        with st.spinner("Analiza CRQ (3 wywołania Claude + analiza chunkingu)..."):
            result = scorer.score_article(
                selected["content"], selected["slug"], selected["title"]
            )

        scorer.close()

        # Gauge charts
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=5,
            specs=[[{"type": "indicator"}] * 5],
            subplot_titles=["Density", "BLUF", "Chunking", "EAV", "OVERALL CRQ"],
        )

        scores = [
            result.information_density,
            result.bluf_compliance,
            result.chunking_quality,
            result.eav_coverage,
            result.overall_crq,
        ]

        for i, score in enumerate(scores):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {"range": [0, 40], "color": "#ff4444"},
                            {"range": [40, 70], "color": "#ffaa00"},
                            {"range": [70, 100], "color": "#44bb44"},
                        ],
                    },
                ),
                row=1, col=i + 1,
            )

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        # Szczegóły
        with st.expander("Information Density"):
            d = result.density_details
            st.markdown(f"**Faktów:** {d.get('facts_count', '?')}")
            if d.get("good_examples"):
                st.markdown("**Dobre przykłady:** " + ", ".join(d["good_examples"][:3]))
            if d.get("filler_examples"):
                st.markdown("**Ogólniki:** " + ", ".join(d["filler_examples"][:3]))

        with st.expander("BLUF compliance"):
            for b in result.bluf_details:
                icon = "✅" if b.get("has_bluf") else "❌"
                st.markdown(f"{icon} **{b.get('header', '?')}**")
                if b.get("recommendation"):
                    st.caption(b["recommendation"])

        with st.expander("Chunking Quality"):
            c = result.chunking_details
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunków", c.get("num_chunks", 0))
            col2.metric("Avg długość", c.get("avg_length", 0))
            col3.metric("Orphans (<100 zn)", c.get("orphan_chunks", 0))

        with st.expander("EAV coverage"):
            e = result.eav_details
            triples = e.get("triples", [])
            st.markdown(f"**Trójek EAV:** {len(triples)}")
            for t in triples[:10]:
                st.text(f"  {t.get('entity', '?')} — {t.get('attribute', '?')} — {t.get('value', '?')}")

        # Rekomendacje
        if result.recommendations:
            st.markdown("---")
            st.subheader("Rekomendacje")
            for rec in result.recommendations:
                st.markdown(f"- {rec}")

        # Zapis
        results_dir.mkdir(parents=True, exist_ok=True)
        crq_file = results_dir / f"crq_{selected['slug']}.json"
        crq_file.write_text(
            json.dumps({
                "slug": result.article_slug,
                "title": result.article_title,
                "information_density": result.information_density,
                "bluf_compliance": result.bluf_compliance,
                "chunking_quality": result.chunking_quality,
                "eav_coverage": result.eav_coverage,
                "overall_crq": result.overall_crq,
                "recommendations": result.recommendations,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        st.caption(f"Zapisano: {crq_file.name}")


def _render_dashboard(artykuly, results_dir):
    """Dashboard z wynikami wszystkich artykułów."""

    crq_files = sorted(results_dir.glob("crq_*.json")) if results_dir.exists() else []

    if not crq_files:
        st.info("Brak wyników CRQ. Przeanalizuj artykuły w zakładce 'Analiza artykułu'.")

        if st.button("📊 Analizuj wszystkie artykuły", type="primary"):
            from src.crq.crq_scorer import CRQScorer

            scorer = CRQScorer()
            progress = st.progress(0)

            for i, art in enumerate(artykuly):
                progress.progress((i + 1) / len(artykuly), text=f"Analiza: {art['title'][:40]}...")
                result = scorer.score_article(art["content"], art["slug"], art["title"])

                results_dir.mkdir(parents=True, exist_ok=True)
                crq_file = results_dir / f"crq_{art['slug']}.json"
                crq_file.write_text(
                    json.dumps({
                        "slug": result.article_slug,
                        "title": result.article_title,
                        "information_density": result.information_density,
                        "bluf_compliance": result.bluf_compliance,
                        "chunking_quality": result.chunking_quality,
                        "eav_coverage": result.eav_coverage,
                        "overall_crq": result.overall_crq,
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            scorer.close()
            st.success("Analiza zakończona!")
            st.rerun()
        return

    # Załaduj wyniki
    import pandas as pd

    rows = []
    for f in crq_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        rows.append(data)

    df = pd.DataFrame(rows)
    display_cols = ["title", "information_density", "bluf_compliance", "chunking_quality", "eav_coverage", "overall_crq"]
    existing_cols = [c for c in display_cols if c in df.columns]

    st.subheader("Porównanie artykułów")
    st.dataframe(
        df[existing_cols].style.background_gradient(
            subset=[c for c in existing_cols if c != "title"],
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Wykres
    if len(rows) > 1:
        import plotly.express as px

        metrics = ["information_density", "bluf_compliance", "chunking_quality", "eav_coverage"]
        existing_metrics = [m for m in metrics if m in df.columns]
        df_melted = df.melt(
            id_vars=["title"], value_vars=existing_metrics,
            var_name="Wymiar", value_name="Score",
        )
        fig = px.bar(
            df_melted, x="title", y="Score", color="Wymiar",
            barmode="group", range_y=[0, 100], height=400,
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
