"""Strona PageIndex — reasoning-based RAG bez embeddingów."""

import json
import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def _wczytaj_artykul(plik: Path) -> tuple[str, str]:
    """Wczytaj tytuł i treść artykułu."""
    tekst = plik.read_text(encoding="utf-8")
    title = plik.stem
    content = tekst
    if tekst.startswith("---"):
        parts = tekst.split("---", 2)
        if len(parts) >= 3:
            import yaml
            try:
                meta = yaml.safe_load(parts[1])
                title = meta.get("title", plik.stem)
            except Exception:
                pass
            content = parts[2].strip()
    return title, content


def render():
    st.title("13. PageIndex (Vectorless RAG)")

    with st.expander("Czym jest PageIndex?", expanded=False):
        st.markdown("""
**Klasyczny RAG:** szuka chunków podobnych do pytania (similarity search).

**PageIndex:** buduje **hierarchiczne drzewo** sekcji artykułu, Claude **rozumuje**
po tym drzewie jak ekspert: "Gdzie szukałby odpowiedzi na to pytanie?"

**Zalety:**
- Brak embeddingów, brak vector DB
- Pełna wyjaśnialność (wiesz dokładnie które sekcje użyte)
- Zachowanie struktury dokumentu (H2/H3)
- Idealne do długich, strukturalnych dokumentów (raporty, umowy)

**Flow:**
1. Budowa drzewa: Claude generuje streszczenie każdej sekcji H2/H3
2. Query-time: Claude dostaje drzewo (tytuły+streszczenia) → wybiera node_id
3. System pobiera pełny tekst TYLKO z wybranych sekcji
4. Claude generuje odpowiedź
        """)

    raw_dir = _root / "data" / "raw"
    pliki = sorted(raw_dir.glob("*.md")) if raw_dir.exists() else []

    if not pliki:
        st.warning("Brak artykułów w data/raw/.")
        return

    # Cache drzew w session_state
    if "pageindex_trees" not in st.session_state:
        st.session_state["pageindex_trees"] = {}

    trees = st.session_state["pageindex_trees"]

    # --- Wybór artykułu ---
    selected_plik = st.selectbox(
        "Wybierz artykuł",
        pliki,
        format_func=lambda p: p.name,
        key="pi_article",
    )

    title, content = _wczytaj_artykul(selected_plik)

    # --- Budowa drzewa ---
    tree_key = selected_plik.name

    if tree_key not in trees:
        st.info("Drzewo PageIndex jeszcze nie zbudowane dla tego artykułu.")
        if st.button("🌳 Zbuduj drzewo", type="primary"):
            from src.pageindex.tree_builder import TreeBuilder

            builder = TreeBuilder()
            with st.spinner("Budowanie drzewa (Claude generuje streszczenia sekcji)..."):
                tree = builder.build_from_markdown(content, title)
            builder.close()

            trees[tree_key] = tree
            st.session_state["pageindex_trees"] = trees
            st.success("Drzewo zbudowane!")
            st.rerun()
    else:
        tree = trees[tree_key]

        # Wizualizacja drzewa
        st.markdown("---")
        st.subheader("Drzewo PageIndex")

        def _render_node(node, indent=0):
            prefix = "  " * indent
            if node.level == 0:
                st.markdown(f"**📄 {node.title}** (root)")
            elif node.level == 2:
                st.markdown(f"{prefix}**##** {node.title}")
                st.caption(f"{prefix}_{node.summary}_")
            elif node.level == 3:
                st.markdown(f"{prefix}**###** {node.title}")
                st.caption(f"{prefix}_{node.summary}_")

            for child in node.children:
                _render_node(child, indent + 1)

        with st.expander(f"Pokaż drzewo ({len(tree.children)} sekcji H2)", expanded=False):
            _render_node(tree)

        # Przycisk do przebudowy
        if st.button("🔄 Przebuduj drzewo"):
            del trees[tree_key]
            st.session_state["pageindex_trees"] = trees
            st.rerun()

        # --- Zapytanie ---
        st.markdown("---")
        st.subheader("Zapytanie")
        query = st.text_input("Pytanie", key="pi_query")

        if st.button("🔍 Nawiguj i odpowiedz", type="primary", disabled=not query):
            from src.pageindex.navigator import TreeNavigator

            navigator = TreeNavigator()

            with st.spinner("Reasoning po drzewie..."):
                result = navigator.navigate(tree, query)

            navigator.close()

            # Wybrane sekcje
            st.markdown("---")
            st.subheader(f"Wybrane sekcje ({len(result.selected_nodes)})")
            for node in result.selected_nodes:
                with st.expander(f"📖 {node.title}"):
                    st.caption(f"node_id: `{node.node_id}` | level: H{node.level}")
                    st.markdown(f"**Streszczenie:** {node.summary}")
                    st.markdown("**Pełny tekst:**")
                    st.text(node.content[:2000] + ("..." if len(node.content) > 2000 else ""))

            # Trace
            with st.expander("Reasoning trace (Claude)"):
                st.code(result.reasoning_trace)

            # Odpowiedź
            st.markdown("---")
            st.subheader("Odpowiedź")
            st.markdown(result.answer)
