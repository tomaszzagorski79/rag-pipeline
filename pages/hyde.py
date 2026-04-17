"""Strona HyDE — Hypothetical Document Embeddings."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.chunking.registry import AVAILABLE_METHODS


def render():
    st.title("6. HyDE (Hypothetical Document Embeddings)")

    with st.expander("Czym jest HyDE?", expanded=False):
        st.markdown("""
**Standardowy RAG:** embed pytania → szukaj podobnych chunków.

**HyDE:** LLM generuje "idealną odpowiedź" → embed tej odpowiedzi → szukaj podobnych chunków.

**Dlaczego działa lepiej?** Hipoteza wygląda jak dokument (długi, ekspercki tekst),
więc embedding hipotezy jest bliższy embeddingom chunków niż embedding krótkiego pytania.

---
**Kiedy używać:**
- Zapytania eksploracyjne, research ("Co warto wiedzieć o cross-border?")
- Nowe domeny bez labeled data
- Query-document gap (pytanie krótkie, odpowiedź długa)

**Kiedy NIE używać:**
- Pytania o precyzyjne fakty/liczby (HyDE halucynuje wartości — Recall@5 = 0.544 vs 0.587 dense)
- Wymagana niska latencja (dodatkowe wywołanie LLM)
- Domeny fact-heavy: finanse, medycyna, prawo
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
        st.warning("Brak kolekcji. Najpierw zaindeksuj artykuły.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        metoda = st.selectbox("Kolekcja", dostepne, key="hyde_method",
                              help="Kolekcja chunków do przeszukania")
    with col2:
        top_k = st.number_input("Top-K", value=5, min_value=1, max_value=20, key="hyde_topk",
                                help="Ile wyników zwrócić z każdego typu search")

    query = st.text_input("Zapytanie", placeholder="Np. Jak zacząć sprzedaż cross-border?", key="hyde_query",
                          help="Pytanie — Claude wygeneruje hipotetyczną odpowiedź i porówna z standardowym search")

    if st.button("🧪 Generuj HyDE i porównaj", type="primary", disabled=not query):
        from src.hyde.hyde_generator import HyDEGenerator
        from src.retrieval.hybrid_search import HybridRetriever
        from src.generation.claude_gen import ClaudeGenerator
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        hyde = HyDEGenerator()
        retriever = HybridRetriever()
        generator = ClaudeGenerator()

        # Standardowy search
        with st.spinner("Standardowy hybrid search..."):
            standard_results = retriever.search(query, collection_name, limit=top_k)

        # HyDE search
        with st.spinner("Generowanie hipotezy i HyDE search..."):
            hypothesis, hyde_results = hyde.search_with_hyde(query, collection_name, limit=top_k)

        # Pokaż hipotezę
        st.markdown("---")
        st.subheader("Wygenerowana hipoteza")
        st.info(hypothesis)

        # Porównanie wyników
        st.markdown("---")
        col_std, col_hyde = st.columns(2)

        with col_std:
            st.subheader("Standardowy search")
            for i, r in enumerate(standard_results, 1):
                st.markdown(f"**[{i}]** Score: `{r.score:.4f}`")
                st.caption(r.metadata.get("slug", ""))
                st.text(r.text[:200] + "...")
                st.markdown("---")

        with col_hyde:
            st.subheader("HyDE search")
            for i, r in enumerate(hyde_results, 1):
                st.markdown(f"**[{i}]** Score: `{r.score:.4f}`")
                st.caption(r.metadata.get("slug", ""))
                st.text(r.text[:200] + "...")
                st.markdown("---")

        # Generuj odpowiedzi z obu zestawów
        st.markdown("---")
        st.subheader("Porównanie odpowiedzi")

        col_ans1, col_ans2 = st.columns(2)

        with col_ans1:
            with st.spinner("Odpowiedź (standard)..."):
                ctx_std = [r.text for r in standard_results]
                ans_std = generator.generate(query, ctx_std)
            st.markdown("**Odpowiedź (standardowy retrieval):**")
            st.markdown(ans_std)

        with col_ans2:
            with st.spinner("Odpowiedź (HyDE)..."):
                ctx_hyde = [r.text for r in hyde_results]
                ans_hyde = generator.generate(query, ctx_hyde)
            st.markdown("**Odpowiedź (HyDE retrieval):**")
            st.markdown(ans_hyde)
