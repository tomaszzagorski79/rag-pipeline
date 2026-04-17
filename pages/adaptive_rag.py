"""Strona Adaptive RAG — dynamiczny routing pytań."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.chunking.registry import AVAILABLE_METHODS


def render():
    st.title("7. Adaptive RAG")

    with st.expander("Czym jest Adaptive RAG?", expanded=False):
        st.markdown("""
**Problem:** Standardowy RAG przepuszcza każde pytanie przez ten sam pipeline.
Proste pytanie nie potrzebuje HyDE + re-ranking. Złożone pytanie potrzebuje więcej kontekstu.

**Adaptive RAG rozwiązuje to:**
- **SIMPLE** → top-3 kontekstów, direct answer (szybko, tanio)
- **MEDIUM** → top-5, standardowy hybrid search
- **COMPLEX** → top-10 + HyDE + re-ranking (pełny pipeline)

Claude klasyfikuje pytanie i wybiera optymalną ścieżkę.
Oszczędność: 30-50% mniej wywołań API przy prostych pytaniach.

---
**Kiedy używać:**
- Zróżnicowane zapytania użytkowników (od prostych do złożonych)
- Optymalizacja kosztów API (proste pytania nie płacą za HyDE+rerank)
- Systemy produkcyjne z dużym wolumenem

**Kiedy NIE używać:**
- Wszystkie pytania mają podobną złożoność
- Potrzebujesz deterministycznego pipeline (classifier może się pomylić)
        """)

    # Sprawdź kolekcje
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()
        dostepne = []
        for method in AVAILABLE_METHODS:
            name = qdrant_cfg.collection_name(method)
            if store.collection_exists(name):
                dostepne.append(method)
        store.close()
    except Exception as e:
        st.error(f"Nie można połączyć z Qdrant: {e}")
        return

    if not dostepne:
        st.warning("Brak kolekcji.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        metoda = st.selectbox("Kolekcja", dostepne, key="adaptive_method",
                              help="Kolekcja Qdrant do przeszukania")
    with col2:
        override = st.selectbox(
            "Override klasyfikacji",
            ["auto", "no_retrieval", "simple", "medium", "complex"],
            key="adaptive_override",
            help="Auto = Claude klasyfikuje. Override = wymuś konkretną ścieżkę.",
        )

    query = st.text_input("Zapytanie", key="adaptive_query")

    # Historia klasyfikacji w session_state
    if "adaptive_history" not in st.session_state:
        st.session_state["adaptive_history"] = []

    if st.button("🧠 Analizuj i odpowiedz", type="primary", disabled=not query):
        from src.adaptive.adaptive_pipeline import AdaptiveRAGPipeline
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        pipeline = AdaptiveRAGPipeline()

        override_val = None if override == "auto" else override

        with st.spinner("Klasyfikacja i retrieval..."):
            result = pipeline.run(query, collection_name, override_classification=override_val)

        # Klasyfikacja
        st.markdown("---")
        colors = {"simple": "green", "medium": "orange", "complex": "red"}
        badges = {"simple": "SIMPLE", "medium": "MEDIUM", "complex": "COMPLEX"}
        color = colors.get(result.classification, "blue")

        st.subheader(f"Klasyfikacja: :{color}[{badges.get(result.classification, '?')}]")
        st.markdown(result.strategy_description)

        # Szczegóły strategii
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Kontekstów", len(result.contexts))
        col_s2.metric("HyDE", "TAK" if result.hyde_hypothesis else "NIE")
        col_s3.metric("Re-ranking", "TAK" if result.was_reranked else "NIE")

        # HyDE hipoteza
        if result.hyde_hypothesis:
            with st.expander("Hipoteza HyDE"):
                st.info(result.hyde_hypothesis)

        # Konteksty
        with st.expander(f"Konteksty ({len(result.results)})"):
            for i, r in enumerate(result.results, 1):
                st.markdown(f"**[{i}]** Score: `{r.score:.4f}`")
                st.caption(r.metadata.get("slug", ""))
                st.text(r.text[:300] + "...")
                st.markdown("---")

        # Odpowiedź
        st.markdown("---")
        st.subheader("Odpowiedź")
        st.markdown(result.answer)

        # Dodaj do historii
        st.session_state["adaptive_history"].append({
            "query": query,
            "classification": result.classification,
        })

    # Pie chart historii klasyfikacji
    history = st.session_state.get("adaptive_history", [])
    if len(history) >= 2:
        st.markdown("---")
        st.subheader(f"Historia klasyfikacji ({len(history)} zapytań)")

        from collections import Counter
        import plotly.express as px

        counts = Counter(h["classification"] for h in history)
        fig_pie = px.pie(
            values=list(counts.values()),
            names=list(counts.keys()),
            title="Rozkład klasyfikacji w tej sesji",
            color_discrete_map={
                "no_retrieval": "#aaaaaa",
                "simple": "#44bb44",
                "medium": "#ffaa00",
                "complex": "#ff4444",
            },
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        if st.button("🗑️ Wyczyść historię"):
            st.session_state["adaptive_history"] = []
            st.rerun()
