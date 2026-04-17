"""Strona Hallucination Detection — weryfikacja twierdzeń w odpowiedziach RAG."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.chunking.registry import AVAILABLE_METHODS


def render():
    st.title("10. Hallucination Detection")

    with st.expander("Czym jest detekcja halucynacji?", expanded=False):
        st.markdown("""
**Halucynacja:** LLM generuje przekonująco brzmiącą, ale **nieprawdziwą** informację.

**Jak to wykrywamy:**
1. Claude **wyciąga twierdzenia faktyczne** z odpowiedzi (np. "VAT w Niemczech to 19%")
2. Każde twierdzenie jest **weryfikowane** osobno względem kontekstów z retrieval
3. Wynik: procent twierdzeń popartych źródłami

**Faithfulness 100%** = wszystkie twierdzenia oparte na kontekście.
**Faithfulness < 70%** = odpowiedź halucynuje — zawiera informacje spoza źródeł.
        """)

    tab_gen, tab_manual = st.tabs(["Generuj i weryfikuj", "Weryfikuj tekst"])

    with tab_gen:
        _render_generate_and_verify()

    with tab_manual:
        _render_manual_verify()


def _render_generate_and_verify():
    """Pełny flow: query → retrieval → generation → verification."""

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
    except Exception:
        st.warning("Nie można połączyć z Qdrant.")
        return

    if not dostepne:
        st.warning("Brak kolekcji.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        metoda = st.selectbox("Kolekcja", dostepne, key="hall_method")
    with col2:
        top_k = st.number_input("Top-K", value=5, min_value=1, max_value=20, key="hall_topk")

    query = st.text_input("Zapytanie", key="hall_query")

    if st.button("🔍 Generuj i weryfikuj", type="primary", disabled=not query):
        from src.retrieval.hybrid_search import HybridRetriever
        from src.generation.claude_gen import ClaudeGenerator
        from src.hallucination.claim_verifier import ClaimVerifier
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        retriever = HybridRetriever()
        generator = ClaudeGenerator()
        verifier = ClaimVerifier()

        # Retrieval + generation
        with st.spinner("Retrieval + generowanie odpowiedzi..."):
            results = retriever.search(query, collection_name, limit=top_k)
            contexts = [r.text for r in results]
            answer = generator.generate(query, contexts)

        st.subheader("Odpowiedź")
        st.markdown(answer)

        # Weryfikacja
        with st.spinner("Weryfikacja twierdzeń (to może chwilę potrwać)..."):
            report = verifier.verify_answer(answer, contexts)

        verifier.close()
        _display_report(report)


def _render_manual_verify():
    """Ręczna weryfikacja — wklej odpowiedź i konteksty."""

    answer = st.text_area("Odpowiedź do weryfikacji", height=200, key="hall_answer")
    contexts_raw = st.text_area(
        "Konteksty (jeden na linię, oddzielone pustą linią)",
        height=200,
        key="hall_contexts",
    )

    if st.button("🔍 Weryfikuj twierdzenia", disabled=not answer or not contexts_raw):
        from src.hallucination.claim_verifier import ClaimVerifier

        contexts = [c.strip() for c in contexts_raw.split("\n\n") if c.strip()]
        if not contexts:
            contexts = [c.strip() for c in contexts_raw.split("\n") if c.strip()]

        verifier = ClaimVerifier()

        with st.spinner("Weryfikacja twierdzeń..."):
            report = verifier.verify_answer(answer, contexts)

        verifier.close()
        _display_report(report)


def _display_report(report):
    """Wyświetl raport z detekcji halucynacji."""
    from src.hallucination.claim_verifier import HallucinationReport

    st.markdown("---")
    st.subheader("Raport halucynacji")

    # Gauge chart
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=report.overall_score,
        title={"text": "Faithfulness Score (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "#ff4444"},
                {"range": [50, 70], "color": "#ffaa00"},
                {"range": [70, 100], "color": "#44bb44"},
            ],
        },
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Metryki
    col1, col2, col3 = st.columns(3)
    col1.metric("Twierdzenia łącznie", len(report.claims))
    col2.metric("Poparte", report.num_supported)
    col3.metric("Niepoparte", report.num_unsupported)

    # Szczegóły per twierdzenie
    st.subheader("Weryfikacja per twierdzenie")
    for i, claim in enumerate(report.claims, 1):
        if claim.supported:
            icon = "✅"
            color = "green"
        else:
            icon = "❌"
            color = "red"

        with st.expander(f"{icon} [{i}] {claim.claim}", expanded=not claim.supported):
            st.markdown(f"**Status:** :{color}[{'POPARTE' if claim.supported else 'NIEPOPARTE'}]")
            st.markdown(f"**Pewność:** {claim.confidence:.0%}")
            st.markdown(f"**Dowód:** {claim.evidence}")
