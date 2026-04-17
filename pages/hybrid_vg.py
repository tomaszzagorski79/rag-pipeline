"""Strona Hybrid RAG (Vector + Graph) — łączenie Qdrant i Neo4j."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("16. Hybrid RAG (Vector + Graph)")

    with st.expander("Czym jest Hybrid Vector + Graph?", expanded=False):
        st.markdown("""
**Połączenie dwóch światów:**
- **Vector DB (Qdrant)** — semantyczne podobieństwo (dense + BM25)
- **Graph DB (Neo4j)** — relacje między encjami (EAV)

**Flow:**
1. Zapytanie trafia **równolegle** do obu źródeł
2. Vector search zwraca top-K chunków (semantyka)
3. Graph search zwraca chunki wspominające encje z pytania (relacje)
4. Claude łączy oba konteksty i generuje odpowiedź

**Przykład:** "Jakie metody płatności są popularne w Niemczech?"
- Vector: znajduje chunki o płatnościach (semantycznie)
- Graph: znajduje chunki linkowane do encji "Niemcy" i "płatności"
- Synteza: pełniejszy kontekst niż pojedyncze źródło.

---
**Kiedy używać:**
- Potrzebujesz ZARÓWNO semantyki (co znaczy pytanie) JAK I relacji (kto z kim)
- E-commerce: produkt + kategoria + marka + atrybuty
- Gdy vector-only lub graph-only nie daje pełnych odpowiedzi

**Kiedy NIE używać:**
- Jedna baza wystarczy (sprawdź RAGAS najpierw)
- Brak Neo4j / graf pusty
        """)

    # Sprawdź Qdrant
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore

        qdrant = QdrantStore()
        qdrant_cfg = get_qdrant_config()
        dostepne = []
        for method in ["naive", "header", "semantic"]:
            name = qdrant_cfg.collection_name(method)
            if qdrant.collection_exists(name):
                dostepne.append(method)
        qdrant.close()
    except Exception as e:
        st.error(f"Qdrant: {e}")
        return

    # Sprawdź Neo4j
    try:
        from src.graph_rag.graph_store import Neo4jStore
        graph = Neo4jStore()
        if not graph.verify():
            st.error("Neo4j: brak połączenia. Sprawdź .env.")
            return
        stats = graph.get_stats()
        graph.close()
    except ValueError as e:
        st.error(f"Neo4j: {e}")
        return

    if stats.get("entities", 0) == 0:
        st.warning("Graf jest pusty. Zbuduj go w zakładce 15. Graph RAG.")
        return

    if not dostepne:
        st.warning("Brak kolekcji Qdrant. Zaindeksuj artykuły.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        metoda = st.selectbox("Kolekcja Qdrant", dostepne, key="hvg_method")
    with col2:
        top_k = st.number_input("Top-K vector", value=5, min_value=1, max_value=15, key="hvg_topk")

    query = st.text_input("Zapytanie", key="hvg_query")

    if st.button("🔀 Hybrid Vector + Graph search", type="primary", disabled=not query):
        from src.retrieval.hybrid_search import HybridRetriever
        from src.graph_rag.graph_retriever import GraphRetriever
        from src.generation.claude_gen import ClaudeGenerator
        from config.settings import get_qdrant_config

        qdrant_cfg = get_qdrant_config()
        collection_name = qdrant_cfg.collection_name(metoda)

        vector_retriever = HybridRetriever()
        graph_retriever = GraphRetriever()
        generator = ClaudeGenerator()

        # 1. Vector search (Qdrant)
        with st.spinner("Vector search (Qdrant)..."):
            vector_results = vector_retriever.search(query, collection_name, limit=top_k)

        # 2. Graph search (Neo4j)
        with st.spinner("Graph search (Neo4j)..."):
            graph_result = graph_retriever.search(query)

        # Wyświetl oba źródła
        col_vec, col_graph = st.columns(2)

        with col_vec:
            st.subheader(f"🔷 Vector ({len(vector_results)})")
            for i, r in enumerate(vector_results, 1):
                st.markdown(f"**[{i}]** `{r.score:.4f}`")
                st.caption(r.metadata.get("slug", ""))
                st.text(r.text[:200] + "...")
                st.markdown("---")

        with col_graph:
            st.subheader(f"🕸️ Graph ({len(graph_result.graph_contexts)} encji)")
            st.caption(f"Encje: {', '.join(graph_result.extracted_entities)}")
            for ctx in graph_result.graph_contexts:
                st.markdown(f"**Encja: {ctx.entity}**")
                for m in ctx.mentions[:3]:
                    attrs = m.get("attrs", [])
                    attrs_str = ", ".join(
                        f"{a.get('attribute')}: {a.get('value')}"
                        for a in attrs if a.get("attribute")
                    )
                    if attrs_str:
                        st.caption(f"↳ {attrs_str}")
                    st.text(m.get("text", "")[:200])
                st.markdown("---")

        # 3. Synteza odpowiedzi z obu źródeł
        st.markdown("---")
        st.subheader("Synteza (Claude)")

        # Buduj kontekst
        all_contexts = []
        for r in vector_results:
            all_contexts.append(f"[Vector] {r.text}")
        for ctx in graph_result.graph_contexts:
            for m in ctx.mentions:
                attrs = m.get("attrs", [])
                attrs_str = "; ".join(
                    f"{a.get('attribute')}={a.get('value')}"
                    for a in attrs if a.get("attribute")
                )
                prefix = f"[Graph:{ctx.entity}"
                if attrs_str:
                    prefix += f" ({attrs_str})"
                prefix += "]"
                all_contexts.append(f"{prefix} {m.get('text', '')}")

        with st.spinner("Generowanie odpowiedzi z hybrydowego kontekstu..."):
            answer = generator.generate(query, all_contexts[:15])

        st.markdown(answer)

        vector_retriever.store.close()
        graph_retriever.close()
