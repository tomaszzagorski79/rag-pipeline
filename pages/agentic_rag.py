"""Strona Agentic RAG — agent Claude z tool_use decyduje jak szukać."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.chunking.registry import AVAILABLE_METHODS


def render():
    st.title("14. Agentic RAG")

    with st.expander("Czym jest Agentic RAG?", expanded=False):
        st.markdown("""
**Różnica od Adaptive RAG:** Adaptive klasyfikuje pytanie raz i wybiera ścieżkę.
**Agentic** — to **pętla ReACT**: Claude sam decyduje co zrobić w każdym kroku.

**Narzędzia agenta:**
- `search_hybrid` — hybrid search (BM25+vector)
- `search_hyde` — HyDE (hipoteza → search)
- `rerank_results` — cross-encoder re-ranking
- `reformulate_query` — przeformułowanie niejasnego pytania

**ReACT loop:** Thought → Action (tool call) → Observation → ... → Final Answer

Agent widzi wyniki narzędzi i decyduje czy potrzebuje więcej informacji, czy może
już odpowiedzieć. Max 5 iteracji.

---
**Kiedy używać:**
- Złożone workflowy, multi-step reasoning
- Pytania wymagające różnych strategii w trakcie (nie wiadomo z góry)
- Research, analiza, due diligence

**Kiedy NIE używać:**
- Proste pytania (5-20x droższe niż Naive RAG: $30-150/1k zapytań)
- Wymagana niska latencja (30-120s per zapytanie)
- Proste FAQ — modular/advanced RAG wystarczy taniej
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

    metoda = st.selectbox("Kolekcja", dostepne, key="agentic_method")
    query = st.text_input("Zapytanie", key="agentic_query")

    if st.button("🤖 Uruchom agenta", type="primary", disabled=not query):
        from src.agentic.agent import AgenticRAG
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        agent = AgenticRAG(collection_name)

        with st.spinner("Agent w toku (może wykonać kilka narzędzi)..."):
            result = agent.run(query)

        agent.close()

        # Metryki
        col1, col2, col3 = st.columns(3)
        col1.metric("Kroków", len(result.steps))
        col2.metric("Wywołań narzędzi", result.total_tool_calls)
        col3.metric("Retrieved chunks", len(result.retrieved_chunks))

        # ReACT trace
        st.markdown("---")
        st.subheader("ReACT Trace")

        for i, step in enumerate(result.steps, 1):
            if step.step_type == "thought":
                with st.expander(f"💭 [{i}] Myśl", expanded=False):
                    st.markdown(step.content)
            elif step.step_type == "tool_call":
                with st.expander(f"🔧 [{i}] Narzędzie: `{step.tool_name}`", expanded=False):
                    st.markdown("**Input:**")
                    st.json(step.tool_input)
                    st.markdown("**Output (skrócony):**")
                    st.text(step.tool_output)
            elif step.step_type == "final_answer":
                st.markdown(f"**✅ [{i}] Final Answer**")

        # Final answer
        st.markdown("---")
        st.subheader("Odpowiedź agenta")
        st.markdown(result.final_answer)

        # Retrieved chunks
        if result.retrieved_chunks:
            with st.expander(f"Ostatnie wyniki retrieval ({len(result.retrieved_chunks)})"):
                for i, r in enumerate(result.retrieved_chunks[:10], 1):
                    st.markdown(f"**[{i}]** Score: `{r.score:.4f}`")
                    st.text(r.text[:300] + "...")
                    st.markdown("---")
