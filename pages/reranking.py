"""Strona Re-ranking — porównanie wyników PRZED i PO re-rankingu cross-encoderem."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("5. Re-ranking (Cross-Encoder)")

    with st.expander("Czym jest re-ranking?", expanded=False):
        st.markdown("""
**Bi-encoder** (etap 1 — hybrid search): embedduje pytanie i dokumenty **osobno**,
porównuje wektory. Szybki, ale płytki.

**Cross-encoder** (etap 2 — re-ranking): analizuje parę (pytanie + fragment) **razem**.
Wolniejszy, ale znacznie dokładniejszy. Decyduje, który fragment jest NAJLEPSZY.

**Flow:** Hybrid search → top-20 kandydatów → Cross-encoder re-ranking → top-5 wyników.

---
**Kiedy używać:**
- Masz dużo chunków i chcesz precyzyjniejszy ranking
- Pytania faktograficzne z konkretnymi terminami
- Zawsze jako pierwszy upgrade do Naive RAG (+39% Recall@5 wg T2-RAGBench)

**Kiedy NIE używać:**
- Bardzo mała baza (<50 chunków) — poprawa minimalna
- Wymagana ultra-niska latencja — cross-encoder dodaje ~200ms
        """)

    # Sprawdź kolekcje
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()
        dostepne = []
        for method in ["naive", "header", "semantic"]:
            name = qdrant_cfg.collection_name(method)
            if store.collection_exists(name):
                dostepne.append(method)
        store.close()
    except Exception as e:
        st.error(f"Nie można połączyć z Qdrant: {e}")
        return

    if not dostepne:
        st.warning("Brak kolekcji. Najpierw zaindeksuj artykuły.")
        return

    # Konfiguracja
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        metoda = st.selectbox("Kolekcja", dostepne, key="rr_method",
                              help="Kolekcja Qdrant z chunkami (naive/header/semantic)")
    with col2:
        prefetch = st.number_input("Prefetch (kandydaci)", value=20, min_value=5, max_value=50, key="rr_prefetch",
                                   help="Ile chunków pobrać z hybrid search PRZED re-rankingiem")
    with col3:
        top_k = st.number_input("Top-K po re-rankingu", value=5, min_value=1, max_value=20, key="rr_topk",
                                help="Ile najlepszych chunków zostawić PO re-rankingu")

    query = st.text_input("Zapytanie", placeholder="Np. Jakie są stawki VAT w UE?", key="rr_query",
                          help="Pytanie które zostanie wyszukane i re-rankowane")

    if st.button("🔍 Szukaj i re-rankuj", type="primary", disabled=not query):
        from src.retrieval.hybrid_search import HybridRetriever
        from src.reranking.flashrank_reranker import FlashRankReranker
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        retriever = HybridRetriever()
        reranker = FlashRankReranker()

        # Retrieval
        with st.spinner("Hybrid search..."):
            raw_results = retriever.search(query, collection_name, limit=prefetch)

        # Re-ranking
        with st.spinner("Re-ranking (cross-encoder)..."):
            reranked = reranker.rerank(query, raw_results, top_k=top_k)

        # Porównanie
        st.markdown("---")
        col_before, col_after = st.columns(2)

        with col_before:
            st.subheader(f"PRZED (hybrid, top-{top_k})")
            for i, r in enumerate(raw_results[:top_k], 1):
                score_color = "🟢" if r.score > 0.5 else "🟡" if r.score > 0.3 else "🔴"
                st.markdown(f"**[{i}]** {score_color} Score: `{r.score:.4f}`")
                st.caption(r.metadata.get("slug", ""))
                st.text(r.text[:200] + "...")
                st.markdown("---")

        with col_after:
            st.subheader(f"PO re-rankingu (top-{top_k})")
            for i, r in enumerate(reranked, 1):
                orig_score = r.metadata.get("original_score", 0)
                st.markdown(f"**[{i}]** Score: `{r.score:.4f}` (było: `{orig_score:.4f}`)")
                st.caption(r.metadata.get("slug", ""))
                st.text(r.text[:200] + "...")
                st.markdown("---")

        # Wykres porównawczy
        if reranked:
            import plotly.graph_objects as go

            fig = go.Figure()
            labels = [f"#{i+1}" for i in range(len(reranked))]
            fig.add_trace(go.Bar(
                name="Po re-rankingu",
                x=labels,
                y=[r.score for r in reranked],
            ))
            fig.add_trace(go.Bar(
                name="Oryginalny score",
                x=labels,
                y=[r.metadata.get("original_score", 0) for r in reranked],
            ))
            fig.update_layout(
                title="Porównanie scorów",
                barmode="group",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
