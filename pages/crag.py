"""Strona CRAG — Corrective RAG z wizualizacją łańcucha decyzji."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.chunking.registry import AVAILABLE_METHODS


def render():
    st.title("8. CRAG (Corrective RAG)")

    with st.expander("Czym jest CRAG?", expanded=False):
        st.markdown("""
**Problem:** Standardowy RAG zawsze generuje odpowiedź — nawet gdy retrieval zwrócił
słabe wyniki. Efekt: halucynacje.

**CRAG rozwiązuje to:**
1. Po retrieval sprawdza **jakość wyników** (score)
2. Jeśli score za niski → **przeformułowuje pytanie** (Claude) i szuka ponownie
3. Jeśli nadal za niski → **odmawia odpowiedzi** zamiast halucynować

**UWAGA:** Score RRF (Qdrant) to ~0.01-0.08, nie 0-1 jak cosine similarity.

---
**Kiedy używać:**
- Jakość źródeł niestabilna (mieszanka dobrych i słabych artykułów)
- Krytyczne zastosowania gdzie halucynacja = problem (finanse, prawo)
- Gdy chcesz "fail gracefully" — odmowa lepiej niż bzdury

**Kiedy NIE używać:**
- Dobra jakość indeksu (hybrid+rerank daje lepszy Recall@5 = 0.816 vs CRAG 0.658)
- Wymagana niska latencja (reformulacja = dodatkowe wywołanie LLM)
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

    # Konfiguracja
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        metoda = st.selectbox("Kolekcja", dostepne, key="crag_method",
                              help="Kolekcja Qdrant z chunkami")
    with col2:
        threshold = st.number_input("Próg score", value=0.02, min_value=0.001, max_value=1.0, step=0.005, format="%.3f", key="crag_thresh",
                                    help="Minimalny score RRF żeby zaakceptować wyniki (RRF: ~0.01-0.08)")
    with col3:
        max_retries = st.number_input("Max prób", value=1, min_value=0, max_value=3, key="crag_retries",
                                      help="Ile razy przeformułować pytanie gdy wyniki poniżej progu")
    with col4:
        top_k = st.number_input("Top-K", value=5, min_value=1, max_value=20, key="crag_topk",
                                help="Ile kontekstów pobrać przy każdej próbie")

    query = st.text_input("Zapytanie", key="crag_query")

    if st.button("🔄 Uruchom CRAG", type="primary", disabled=not query):
        from src.crag.corrective_rag import CorrectiveRAG
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        crag = CorrectiveRAG(score_threshold=threshold, max_retries=max_retries)

        with st.spinner("CRAG w toku..."):
            result = crag.run(query, collection_name, limit=top_k)

        # Wizualizacja łańcucha decyzji
        st.markdown("---")
        st.subheader("Łańcuch decyzji")

        for step in result.steps:
            if step.decision == "accept":
                color = "green"
                icon = "✅"
                desc = "Wyniki zaakceptowane"
            elif step.decision == "reformulate":
                color = "orange"
                icon = "🔄"
                desc = "Przeformułowanie pytania"
            else:
                color = "red"
                icon = "❌"
                desc = "Odrzucenie — za słabe wyniki"

            with st.expander(f"{icon} {step.step_name}: {desc}", expanded=True):
                st.markdown(f"**Zapytanie:** {step.query_used}")
                st.markdown(f"**Najlepszy score:** `{step.best_score:.4f}` (próg: `{threshold:.3f}`)")
                st.markdown(f"**Wyników:** {step.num_results}")
                st.markdown(f"**Decyzja:** :{color}[{step.decision.upper()}]")

        # Status
        if result.was_reformulated:
            st.info("Pytanie zostało przeformułowane.")
        if result.was_rejected:
            st.error("CRAG odrzucił wyniki — brak wiarygodnych danych.")

        # Odpowiedź
        st.markdown("---")
        st.subheader("Odpowiedź")
        st.markdown(result.final_answer)

        # Konteksty
        if result.final_results:
            with st.expander(f"Konteksty ({len(result.final_results)})"):
                for i, r in enumerate(result.final_results, 1):
                    st.markdown(f"**[{i}]** Score: `{r.score:.4f}`")
                    st.text(r.text[:300] + "...")
                    st.markdown("---")
