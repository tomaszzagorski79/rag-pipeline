"""Strona RAG-Fusion — multi-query + RRF fusion."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("18. RAG-Fusion")

    with st.expander("Czym jest RAG-Fusion?", expanded=False):
        st.markdown("""
**Problem:** Jedno pytanie eksploruje tylko jedną perspektywę.

**RAG-Fusion:**
1. Claude generuje **N wariantów** pytania (różne perspektywy, słowa kluczowe)
2. Każdy wariant → **osobny retrieval** (top-K)
3. Wyniki łączone przez **RRF** (Reciprocal Rank Fusion)
4. Chunki które pojawiają się w wielu wariantach → wyższy ranking

**Efekt:** +22% NDCG, +40% recall vs single-query (wg benchmarku).

---
**Kiedy używać:**
- Pytania szerokie, eksploracyjne, research
- Chcesz pokryć różne aspekty tematu jednym zapytaniem
- Wiele perspektyw na to samo pytanie

**Kiedy NIE używać:**
- Pytania wąskie/faktologiczne (dodaje szum)
- Wymagana niska latencja (N × retrieval = ~1.7x czas)
- Budżet ograniczony (dodatkowe wywołanie LLM na sub-queries)
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
        metoda = st.selectbox("Kolekcja", dostepne, key="rf_method",
                              help="Kolekcja Qdrant z chunkami")
    with col2:
        n_queries = st.number_input("Wariantów pytania", value=4, min_value=2, max_value=8, key="rf_n",
                                    help="Ile wariantów pytania wygenerować (Claude). Więcej = lepszy recall, ale droższe.")
    with col3:
        top_k = st.number_input("Top-K final", value=5, min_value=1, max_value=15, key="rf_k",
                                help="Ile wyników po RRF fusion")

    query = st.text_input("Zapytanie", key="rf_query")

    if st.button("🔀 RAG-Fusion", type="primary", disabled=not query):
        from src.rag_fusion.fusion import RAGFusion
        from src.generation.claude_gen import ClaudeGenerator
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        fusion = RAGFusion(n_queries=n_queries)
        generator = ClaudeGenerator()

        with st.spinner(f"Generowanie {n_queries} wariantów + retrieval każdego..."):
            result = fusion.run(query, collection_name, top_k_final=top_k)

        fusion.close()

        # Sub-queries
        st.markdown("---")
        st.subheader(f"Wygenerowane warianty ({len(result.sub_queries)})")
        for i, sq in enumerate(result.sub_queries):
            label = "🔵 Oryginalne" if i == 0 else f"🟢 Wariant {i}"
            hits = len(result.per_query_results.get(sq, []))
            st.markdown(f"{label}: _{sq}_ ({hits} wyników)")

        # Fused results
        st.markdown("---")
        st.subheader(f"Wyniki po RRF Fusion (top-{top_k})")

        for i, r in enumerate(result.fused_results, 1):
            n_appearances = r.metadata.get("appeared_in_n_queries", 1)
            bar = "🟢" * n_appearances + "⬜" * (len(result.sub_queries) - n_appearances)
            st.markdown(f"**[{i}]** RRF: `{r.score:.4f}` | Pojawił się w {n_appearances}/{len(result.sub_queries)} zapytaniach {bar}")
            st.caption(r.metadata.get("slug", ""))
            st.text(r.text[:300] + "...")
            st.markdown("---")

        # Odpowiedź
        st.subheader("Odpowiedź")
        contexts = [r.text for r in result.fused_results]
        with st.spinner("Generowanie odpowiedzi..."):
            answer = generator.generate(query, contexts)
        st.markdown(answer)
