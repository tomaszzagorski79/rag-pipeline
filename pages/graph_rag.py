"""Strona Graph RAG — retrieval oparty na grafie wiedzy EAV w Neo4j."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("15. Graph RAG")

    with st.expander("Czym jest Graph RAG?", expanded=False):
        st.markdown("""
**Klasyczny RAG:** szuka chunków podobnych semantycznie (vector).

**Graph RAG:** szuka po **relacjach między encjami** w grafie wiedzy.

**Flow:**
1. **Indexing (offline):** Claude wyciąga trójki EAV (Entity-Attribute-Value) z chunków
2. Zapisanie do Neo4j: `(Entity)-[HAS_ATTRIBUTE]->(Value)`, `(Chunk)-[MENTIONS]->(Entity)`
3. **Query-time:** Claude wyciąga encje z pytania → graf zwraca chunki wspominające te encje

**Przewagi:**
- Naturalne dla faktów strukturalnych ("Jakie stawki VAT ma Niemcy?")
- Linkuje rozproszone fakty po tej samej encji
- Wizualne w Neo4j Browser

**Wymaga:** NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD w .env.
Free tier: [Neo4j Aura](https://console.neo4j.io/) (1 instancja, 50k nodes).

---
**Kiedy używać:**
- Multi-hop QA ("Jakie firmy z Niemiec oferują PayPal?")
- Pytania o relacje między encjami (kto, co, gdzie, ile)
- Pytania globalne ("Jakie są główne wątki w korpusie?")
- Topical authority / knowledge graphs w SEO

**Kiedy NIE używać:**
- Mały korpus (<10 artykułów) — overhead budowy grafu się nie opłaca
- Pytania czysto semantyczne bez encji (lepiej: vector hybrid)
- Dane bez wyraźnych encji i relacji
        """)

    # Sprawdź połączenie
    try:
        from src.graph_rag.graph_store import Neo4jStore
        store = Neo4jStore()
        if not store.verify():
            st.error("Nie można połączyć z Neo4j. Sprawdź NEO4J_URI i NEO4J_PASSWORD w .env.")
            return
    except ValueError as e:
        st.error(str(e))
        st.info("""
**Setup Neo4j Aura (free):**
1. Wejdź na https://console.neo4j.io/
2. Utwórz AuraDB Free instance
3. Pobierz credentials
4. Dodaj do .env:
```
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```
        """)
        return
    except Exception as e:
        st.error(f"Błąd połączenia: {e}")
        return

    # --- Statystyki grafu ---
    stats = store.get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Encje", stats.get("entities", 0))
    col2.metric("Chunki", stats.get("chunks", 0))
    col3.metric("Atrybuty", stats.get("attributes", 0))
    col4.metric("Wzmianki", stats.get("mentions", 0))

    store.close()

    # --- Budowa grafu ---
    st.markdown("---")
    st.header("Budowa grafu EAV")

    col_build, col_clear = st.columns(2)

    with col_build:
        if st.button("🔨 Zbuduj graf z artykułów", type="primary"):
            _build_graph()

    with col_clear:
        if st.button("🗑️ Wyczyść graf"):
            from src.graph_rag.graph_store import Neo4jStore
            s = Neo4jStore()
            s.clear_graph()
            s.close()
            st.success("Graf wyczyszczony.")
            st.rerun()

    # --- Top encje ---
    if stats.get("entities", 0) > 0:
        st.markdown("---")
        st.subheader("Top encje w grafie")

        from src.graph_rag.graph_store import Neo4jStore
        s = Neo4jStore()
        top = s.get_top_entities(limit=15)
        s.close()

        if top:
            import pandas as pd
            df = pd.DataFrame(top)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Query ---
    st.markdown("---")
    st.header("Zapytanie w Graph RAG")

    query = st.text_input(
        "Pytanie",
        placeholder="Np. Jakie są stawki VAT w UE?",
        key="graph_query",
    )

    if st.button("🔍 Szukaj w grafie", disabled=not query):
        from src.graph_rag.graph_retriever import GraphRetriever

        retriever = GraphRetriever()

        with st.spinner("Ekstrakcja encji + szukanie w grafie..."):
            result = retriever.search(query)

        retriever.close()

        # Wyciągnięte encje
        st.markdown(f"**Wyciągnięte encje:** {', '.join(result.extracted_entities)}")

        if not result.graph_contexts:
            st.warning("Brak encji znalezionych w grafie. Spróbuj innych słów kluczowych.")
            return

        # Konteksty per encja
        for ctx in result.graph_contexts:
            with st.expander(f"🏷️ Encja: **{ctx.entity}** ({len(ctx.mentions)} wzmianek)"):
                for m in ctx.mentions:
                    st.markdown(f"**Chunk:** `{m.get('chunk_id', '?')[:30]}`")
                    st.caption(f"Artykuł: {m.get('slug', '?')}")

                    # Atrybuty
                    attrs = m.get("attrs", [])
                    if attrs:
                        for a in attrs:
                            if a.get("attribute") and a.get("value"):
                                st.markdown(f"  • **{a['attribute']}**: {a['value']}")

                    st.text(m.get("text", "")[:300])
                    st.markdown("---")


def _build_graph():
    """Buduje graf z istniejących kolekcji Qdrant."""
    from config.settings import get_qdrant_config
    from src.vectorstore.qdrant_store import QdrantStore
    from src.graph_rag.eav_extractor import EAVExtractor
    from src.graph_rag.graph_store import Neo4jStore

    qdrant_cfg = get_qdrant_config()
    qdrant = QdrantStore()

    # Użyj kolekcji header (najczystsza struktura)
    collection = qdrant_cfg.collection_name("header")
    if not qdrant.collection_exists(collection):
        st.error(f"Kolekcja {collection} nie istnieje. Najpierw zaindeksuj artykuły (header).")
        return

    # Pobierz chunki
    from qdrant_client import models
    scroll_result, _ = qdrant.client.scroll(
        collection_name=collection,
        limit=1000,
        with_payload=True,
    )

    chunks_data = []
    for point in scroll_result:
        payload = point.payload or {}
        chunks_data.append({
            "chunk_id": payload.get("chunk_id", str(point.id)),
            "text": payload.get("text", ""),
            "slug": payload.get("slug", ""),
        })

    st.info(f"Przetwarzanie {len(chunks_data)} chunków...")

    # Ekstrakcja EAV + upsert do Neo4j
    extractor = EAVExtractor()
    graph = Neo4jStore()

    progress = st.progress(0, text="Ekstrakcja EAV...")
    total_triples = 0

    for i, chunk in enumerate(chunks_data):
        progress.progress(
            (i + 1) / len(chunks_data),
            text=f"Chunk {i+1}/{len(chunks_data)}: ekstrakcja EAV i upsert do grafu...",
        )

        triples = extractor.extract(chunk["text"])
        if triples:
            graph.upsert_eav_triples(
                chunk_id=chunk["chunk_id"],
                chunk_text=chunk["text"],
                slug=chunk["slug"],
                triples=triples,
            )
            total_triples += len(triples)

    extractor.close()
    graph.close()
    qdrant.close()

    st.success(f"Zbudowano graf: {total_triples} trójek EAV z {len(chunks_data)} chunków.")
    st.rerun()
