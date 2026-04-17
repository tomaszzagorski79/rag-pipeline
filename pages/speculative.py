"""Strona Speculative RAG — drafter + verifier."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.chunking.registry import AVAILABLE_METHODS


def render():
    st.title("21. Speculative RAG")

    with st.expander("Czym jest Speculative RAG?", expanded=False):
        st.markdown("""
**Standardowy RAG:** jeden model, jedna odpowiedź.

**Speculative RAG:**
1. **Drafter** (Claude Haiku — szybki, tani) generuje **N draftów** z różnych podzbiorów kontekstu
2. **Verifier** (Claude Sonnet — dokładny) ocenia wszystkie drafty: faithfulness, relevance, completeness
3. Najlepszy draft = finalna odpowiedź

**Efekt:** 51% szybciej (Google benchmark), wyższa jakość bo porównujemy N opcji.

**Analogia:** jak burza mózgów — kilka szybkich propozycji, potem ekspert wybiera najlepszą.

---
**Kiedy używać:**
- Potrzebujesz szybszej odpowiedzi (drafter Haiku jest tani i szybki)
- Chcesz porównać kilka perspektyw odpowiedzi
- Koszt nie jest problemem (drafter + verifier = 2 modele)

**Kiedy NIE używać:**
- Proste pytania (overhead 2 modeli nie opłaca się)
- Domeny wymagające max dokładności (drafter może systematycznie popełniać błędy)
- Ograniczony budżet API
        """)

    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore
        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()
        dostepne = [m for m in AVAILABLE_METHODS
                    if store.collection_exists(qdrant_cfg.collection_name(m))]
        store.close()
    except Exception as e:
        st.error(f"Qdrant: {e}")
        return

    if not dostepne:
        st.warning("Brak kolekcji.")
        return

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        metoda = st.selectbox("Kolekcja", dostepne, key="spec_method",
                              help="Kolekcja chunków z Qdrant")
    with col2:
        n_drafts = st.number_input("Liczba draftów", value=3, min_value=2, max_value=5, key="spec_n",
                                   help="Ile draftów wygeneruje Drafter (Haiku). Więcej = lepsze porównanie, ale droższe.")
    with col3:
        top_k = st.number_input("Top-K kontekstów", value=10, min_value=3, max_value=20, key="spec_k",
                                help="Ile chunków pobrać z retrieval (dzielone między drafty)")

    query = st.text_input("Zapytanie", key="spec_query")

    if st.button("⚡ Speculative RAG", type="primary", disabled=not query):
        from src.retrieval.hybrid_search import HybridRetriever
        from src.speculative.speculative_rag import SpeculativeRAG
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        retriever = HybridRetriever()
        spec = SpeculativeRAG(n_drafts=n_drafts)

        # Retrieval
        with st.spinner("Retrieval..."):
            results = retriever.search(query, collection_name, limit=top_k)
            contexts = [r.text for r in results]

        # Speculative
        with st.spinner(f"Drafter (Haiku) generuje {n_drafts} draftów + Verifier (Sonnet) ocenia..."):
            result = spec.run(query, contexts)

        spec.close()
        retriever.store.close()

        # Drafty z ocenami
        st.markdown("---")
        st.subheader(f"Drafty ({len(result.drafts)})")

        for draft in result.drafts:
            icon = "🏆" if draft.is_selected else "📝"
            color = "green" if draft.is_selected else "gray"

            with st.expander(
                f"{icon} Draft {draft.draft_id} — Total: {draft.total_score}/30"
                + (" ← WYBRANY" if draft.is_selected else ""),
                expanded=draft.is_selected,
            ):
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Faithfulness", f"{draft.faithfulness}/10")
                col_s2.metric("Relevance", f"{draft.relevance}/10")
                col_s3.metric("Completeness", f"{draft.completeness}/10")

                st.markdown(draft.text)
                st.caption(f"Kontekstów użyto: {len(draft.context_subset)}")

        # Finalna odpowiedź
        st.markdown("---")
        st.subheader("Finalna odpowiedź (najlepszy draft)")
        st.markdown(result.final_answer)

        # Wykres porównawczy
        import plotly.graph_objects as go

        fig = go.Figure()
        for draft in result.drafts:
            fig.add_trace(go.Bar(
                name=f"Draft {draft.draft_id}" + (" ★" if draft.is_selected else ""),
                x=["Faithfulness", "Relevance", "Completeness"],
                y=[draft.faithfulness, draft.relevance, draft.completeness],
            ))
        fig.update_layout(barmode="group", title="Oceny draftów", height=350, yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)
