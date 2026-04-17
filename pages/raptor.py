"""Strona RAPTOR — hierarchiczny indeks drzewiasty."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("20. RAPTOR")

    with st.expander("Czym jest RAPTOR?", expanded=False):
        st.markdown("""
**RAPTOR** = Recursive Abstractive Processing for Tree-Organized Retrieval.

**Jak działa (indexing):**
1. Chunki = liście drzewa (level 0)
2. Klastry embeddingów (cosine similarity) → grupy
3. Claude generuje **streszczenie** każdej grupy → nowy węzeł (level 1)
4. Powtórz rekursywnie aż do roota

**Jak działa (retrieval):**
- **Collapsed tree:** flat search po WSZYSTKICH węzłach (liście + streszczenia)
- **Tree traversal:** schodzenie od roota

**vs PageIndex:** RAPTOR używa embeddingów, PageIndex używa reasoning.
**vs GraphRAG:** RAPTOR buduje hierarchię tematyczną, GraphRAG buduje graf encji.

---
**Kiedy używać:**
- Długie dokumenty z naturalną hierarchią (podręczniki, raporty roczne)
- Multi-hop QA wymagające różnych poziomów abstrakcji
- Pytania o podsumowanie ("Jakie są główne wnioski?")

**Kiedy NIE używać:**
- Mały korpus (<10 chunków) — drzewo będzie trywialne
- Domeny bez hierarchicznej struktury tematycznej
- Bardzo kosztowna indeksacja (LLM per klaster)
        """)

    raw_dir = _root / "data" / "raw"
    pliki = sorted(raw_dir.glob("*.md")) if raw_dir.exists() else []

    if not pliki:
        st.warning("Brak artykułów.")
        return

    if "raptor_trees" not in st.session_state:
        st.session_state["raptor_trees"] = {}

    trees = st.session_state["raptor_trees"]

    # --- Budowa drzewa ---
    selected = st.selectbox("Artykuł", pliki, format_func=lambda p: p.name, key="raptor_article")

    col1, col2 = st.columns(2)
    with col1:
        cluster_size = st.number_input("Rozmiar klastra", value=5, min_value=2, max_value=10, key="raptor_cs")
    with col2:
        max_levels = st.number_input("Max poziomów", value=3, min_value=1, max_value=5, key="raptor_ml")

    tree_key = f"{selected.name}_{cluster_size}_{max_levels}"

    if tree_key not in trees:
        if st.button("🌲 Zbuduj drzewo RAPTOR", type="primary"):
            from src.raptor.raptor_index import RAPTORBuilder
            from src.chunking.header_based import HeaderChunker

            # Wczytaj i chunk artykuł
            tekst = selected.read_text(encoding="utf-8")
            if tekst.startswith("---"):
                parts = tekst.split("---", 2)
                if len(parts) >= 3:
                    tekst = parts[2].strip()

            chunker = HeaderChunker()
            chunks = chunker.chunk(tekst, {"slug": selected.stem})
            texts = [c.text for c in chunks]

            if len(texts) < 3:
                st.error("Za mało chunków do budowy drzewa RAPTOR.")
                return

            builder = RAPTORBuilder(cluster_size=cluster_size, max_levels=max_levels)
            progress = st.progress(0)

            def update(p, t):
                progress.progress(p, text=t)

            with st.spinner("Budowanie drzewa RAPTOR..."):
                tree = builder.build(texts, progress_callback=update)

            builder.close()
            trees[tree_key] = tree
            st.session_state["raptor_trees"] = trees
            st.success(f"Drzewo zbudowane: {len(tree.nodes)} węzłów, {tree.levels} poziomów")
            st.rerun()
    else:
        tree = trees[tree_key]

        # Statystyki
        st.markdown("---")
        st.subheader("Drzewo RAPTOR")

        cols = st.columns(tree.levels + 2)
        for lvl in range(tree.levels + 1):
            nodes_at_level = tree.get_level(lvl)
            label = "Liście" if lvl == 0 else f"Level {lvl}"
            cols[lvl].metric(label, len(nodes_at_level))
        cols[-1].metric("Total", len(tree.nodes))

        # Treemap drzewa
        st.markdown("---")
        st.subheader("Treemap drzewa RAPTOR")

        import plotly.graph_objects as go

        tm_labels = []
        tm_parents = []
        tm_values = []
        tm_texts = []

        for node_id, node in tree.nodes.items():
            tm_labels.append(node_id)
            # Parent to node który ma ten node_id w children_ids
            parent_id = ""
            for other_id, other in tree.nodes.items():
                if node_id in other.children_ids:
                    parent_id = other_id
                    break
            tm_parents.append(parent_id)
            tm_values.append(max(1, len(node.text) // 10))
            label = f"L{node.level}: {node.text[:60]}..." if node.is_summary else node.text[:60]
            tm_texts.append(label)

        fig_tm = go.Figure(go.Treemap(
            labels=tm_labels,
            parents=tm_parents,
            values=tm_values,
            text=tm_texts,
            hovertext=tm_texts,
            root_color="lightgrey",
            marker=dict(
                colors=[node.level for node in tree.nodes.values()],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Level"),
            ),
        ))
        fig_tm.update_layout(title="Hierarchia RAPTOR — wielkość = długość tekstu", height=500)
        st.plotly_chart(fig_tm, use_container_width=True)

        # Pokaż streszczenia wyższych poziomów
        for lvl in range(1, tree.levels + 1):
            with st.expander(f"Streszczenia Level {lvl}"):
                for node in tree.get_level(lvl):
                    st.markdown(f"**{node.node_id}** ({len(node.children_ids)} dzieci)")
                    st.text(node.text[:500])
                    st.markdown("---")

        if st.button("🔄 Przebuduj"):
            del trees[tree_key]
            st.rerun()

        # --- Retrieval ---
        st.markdown("---")
        st.subheader("Retrieval w RAPTOR")

        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            query = st.text_input("Zapytanie", key="raptor_query")
        with col_q2:
            search_mode = st.radio("Tryb", ["collapsed", "tree"], key="raptor_mode", horizontal=True)

        if st.button("🔍 Szukaj w RAPTOR", disabled=not query):
            from src.raptor.raptor_index import RAPTORBuilder
            from src.generation.claude_gen import ClaudeGenerator

            builder = RAPTORBuilder()

            with st.spinner("Retrieval..."):
                results = builder.search(tree, query, top_k=5, mode=search_mode)

            builder.close()

            st.markdown(f"**Tryb:** {search_mode} | **Wyniki:** {len(results)}")

            for i, node in enumerate(results, 1):
                level_label = "📄 Chunk" if node.level == 0 else f"📊 Summary L{node.level}"
                with st.expander(f"[{i}] {level_label}: {node.node_id}"):
                    st.text(node.text[:1000])

            # Generuj odpowiedź
            generator = ClaudeGenerator()
            contexts = [n.text for n in results]
            with st.spinner("Generowanie odpowiedzi..."):
                answer = generator.generate(query, contexts)

            st.markdown("---")
            st.subheader("Odpowiedź")
            st.markdown(answer)
