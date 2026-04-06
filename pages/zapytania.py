"""Strona zapytań — interaktywne testowanie pipeline'u RAG."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("3. Zapytania")
    st.markdown("Zadaj pytanie i porównaj odpowiedzi z różnych metod chunkingu.")

    # --- Sprawdź dostępne kolekcje ---
    dostepne_metody = []
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()

        for method in ["naive", "header", "semantic"]:
            name = qdrant_cfg.collection_name(method)
            if store.collection_exists(name):
                count = store.count_points(name)
                dostepne_metody.append((method, count))
        store.close()
    except Exception as e:
        st.error(f"Nie można połączyć z Qdrant: {e}")
        return

    if not dostepne_metody:
        st.warning("Brak zaindeksowanych kolekcji. Najpierw uruchom Chunking & Indeksowanie.")
        return

    # Pokaż status
    cols = st.columns(len(dostepne_metody))
    for i, (method, count) in enumerate(dostepne_metody):
        cols[i].metric(f"articles_{method}", f"{count} punktów")

    # --- Konfiguracja wyszukiwania ---
    st.markdown("---")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected_methods = st.multiselect(
            "Porównaj metody",
            [m for m, _ in dostepne_metody],
            default=[m for m, _ in dostepne_metody],
        )
    with col2:
        top_k = st.number_input("Top-K kontekstów", min_value=1, max_value=20, value=5)
    with col3:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05, help="Pokaż tylko chunki z score powyżej tego progu")

    # --- Pytanie ---
    query = st.text_input(
        "Zadaj pytanie",
        placeholder="Np. Jakie są najlepsze praktyki SEO dla sklepów internetowych?",
    )

    if st.button("🔍 Szukaj i generuj odpowiedź", type="primary", disabled=not query):
        from src.retrieval.hybrid_search import HybridRetriever
        from src.generation.claude_gen import ClaudeGenerator
        from config.settings import get_qdrant_config

        retriever = HybridRetriever()
        generator = ClaudeGenerator()
        qdrant_cfg = get_qdrant_config()

        for metoda in selected_methods:
            st.markdown("---")
            st.subheader(f"Metoda: {metoda.upper()}")

            collection_name = qdrant_cfg.collection_name(metoda)

            # Retrieval
            with st.spinner(f"Wyszukiwanie ({metoda})..."):
                results = retriever.search(query, collection_name, limit=top_k)

            if not results:
                st.warning("Brak wyników.")
                continue

            # Filtruj po min score
            filtered = [r for r in results if r.score >= min_score]

            if not filtered:
                st.warning(f"Brak wyników ze score >= {min_score:.2f} (najwyższy: {results[0].score:.4f})")
                continue

            # Konteksty
            label = f"📄 Znalezione konteksty ({len(filtered)}"
            if len(filtered) < len(results):
                label += f", odfiltrowano {len(results) - len(filtered)} poniżej {min_score:.2f}"
            label += ")"

            with st.expander(label, expanded=False):
                for i, r in enumerate(filtered, 1):
                    score_color = "🟢" if r.score > 0.5 else "🟡" if r.score > 0.3 else "🔴"
                    st.markdown(f"**[{i}]** {score_color} Score: `{r.score:.4f}`")

                    source = r.metadata.get("slug", "?")
                    h2 = r.metadata.get("h2", "")
                    info = f"Źródło: {source}"
                    if h2:
                        info += f" | Sekcja: {h2}"
                    st.caption(info)

                    st.text(r.text[:500] + ("..." if len(r.text) > 500 else ""))
                    st.markdown("---")

            # Generation — tylko z przefiltrowanych kontekstów
            with st.spinner(f"Generowanie odpowiedzi ({metoda})..."):
                contexts = [r.text for r in filtered]
                answer = generator.generate(query, contexts)

            st.markdown(answer)

    # --- Historia zapytań ---
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if query and st.session_state.get("_last_query") != query:
        st.session_state.query_history.append(query)
        st.session_state._last_query = query

    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("Historia zapytań")
        for i, q in enumerate(reversed(st.session_state.query_history[-10:]), 1):
            st.text(f"  {i}. {q}")
