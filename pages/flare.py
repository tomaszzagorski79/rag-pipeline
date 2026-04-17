"""Strona FLARE — Forward-Looking Active Retrieval."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("19. FLARE (Active Retrieval)")

    with st.expander("Czym jest FLARE?", expanded=False):
        st.markdown("""
**Standardowy RAG:** szukaj raz → generuj całą odpowiedź.

**FLARE:** generuj **zdanie po zdaniu**. Po każdym zdaniu sprawdź pewność:
- Pewność wysoka → kontynuuj
- Pewność niska → **przerwij, szukaj nowy kontekst**, regeneruj zdanie

**Efekt:** odpowiedź jest groundowana w każdym zdaniu, nie tylko na początku.
Idealne do długich odpowiedzi gdzie kontekst się zmienia w trakcie.

**Uwaga:** wolniejsze (wiele wywołań LLM), ale znacznie mniej halucynacji w długich odpowiedziach.

---
**Kiedy używać:**
- Długie odpowiedzi (long-form) gdzie kontekst się zmienia w trakcie
- Multi-hop reasoning (każdy krok potrzebuje innej wiedzy)
- Max faithfulness — groundowanie per zdanie

**Kiedy NIE używać:**
- Krótkie odpowiedzi (1-2 zdania) — overhead nieproporcjonalny
- Wymagana niska latencja (wielokrotne retrieval+LLM w trakcie generacji)
- Proste FAQ
        """)

    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore
        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()
        dostepne = [m for m in ["naive", "header", "semantic"]
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
        metoda = st.selectbox("Kolekcja", dostepne, key="flare_method",
                              help="Kolekcja Qdrant z chunkami do przeszukania")
    with col2:
        threshold = st.slider("Próg pewności (1-10)", 1, 10, 5, key="flare_thresh",
                              help="Poniżej tego progu Claude przerwie generację i pobierze nowy kontekst")
    with col3:
        max_sent = st.number_input("Max zdań", value=8, min_value=3, max_value=15, key="flare_max",
                                   help="Maksymalna liczba zdań w odpowiedzi (każde osobne wywołanie LLM)")

    query = st.text_input("Zapytanie", key="flare_query")

    if st.button("⚡ FLARE — generuj z aktywnym retrieval", type="primary", disabled=not query):
        from src.flare.flare_generator import FLAREGenerator
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        flare = FLAREGenerator(confidence_threshold=threshold, max_sentences=max_sent)

        with st.spinner("FLARE w toku (generacja zdanie po zdaniu)..."):
            result = flare.run(query, collection_name)

        flare.close()

        # Metryki
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Zdań", len(result.steps))
        col_m2.metric("Active retrievals", result.total_retrievals)
        col_m3.metric("Avg pewność", f"{sum(s.confidence for s in result.steps)/max(len(result.steps),1):.1f}/10")

        # Trace — zdanie po zdaniu
        st.markdown("---")
        st.subheader("FLARE Trace (zdanie po zdaniu)")

        for i, step in enumerate(result.steps, 1):
            if step.confidence >= threshold:
                icon = "🟢"
                color = "green"
            else:
                icon = "🔴"
                color = "red"

            with st.expander(
                f"{icon} [{i}] Pewność: {step.confidence}/10"
                + (" → RETRIEVAL" if step.needed_retrieval else ""),
                expanded=step.needed_retrieval,
            ):
                st.markdown(f"**Zdanie:** {step.sentence}")
                st.markdown(f"**Pewność:** :{color}[{step.confidence}/10]")

                if step.needed_retrieval:
                    st.markdown(f"**Query do retrieval:** _{step.retrieval_query[:100]}_")
                    st.markdown(f"**Nowe konteksty:** {len(step.new_contexts)}")
                    if step.regenerated:
                        st.success("Zdanie zregenerowane z nowym kontekstem")

        # Finalna odpowiedź
        st.markdown("---")
        st.subheader("Finalna odpowiedź")
        st.markdown(result.final_answer)
