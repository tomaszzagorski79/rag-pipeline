"""Strona Context Engineering — pełny framework zarządzania kontekstem LLM."""

import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.chunking.registry import AVAILABLE_METHODS


def render():
    st.title("17. Context Engineering")

    with st.expander("Czym jest Context Engineering?", expanded=False):
        st.markdown("""
**RAG** odpowiada na pytanie "co pobrać?".
**Context Engineering** odpowiada na "jak zarządzać CAŁYM stanem informacyjnym LLM?".

**6 komponentów (wszystkie zaimplementowane):**
1. ✅ **Agent/orkiestrator** — Claude decyduje które źródła wywołać
2. ✅ **Query Augmentation** — rewriting, fan-out, klasyfikacja intencji
3. ✅ **Retrieval wieloźródłowy** — Qdrant + Neo4j + PageIndex równolegle
4. ✅ **Context Window Budgeting** — dzielenie tokenów między komponenty
5. ✅ **Memory wielowarstwowa** — short-term (sesja) + episodic (persystentna)
6. ✅ **Faceted Search** — "peryferyjne widzenie" — co jeszcze jest dostępne

**Na tej stronie:**
- **Auto mode** — orkiestrator sam decyduje wszystko
- **Manual mode** — ty wybierasz źródła i parametry
        """)

    # Cache memory w session_state
    if "context_eng_memory" not in st.session_state:
        from src.context_eng.memory import MultiLayerMemory
        st.session_state["context_eng_memory"] = MultiLayerMemory()

    memory = st.session_state["context_eng_memory"]

    # --- Sprawdź dostępność źródeł ---
    sources = {"vector": False, "graph": False, "pageindex": False}

    dostepne_vec = []
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore
        qdrant = QdrantStore()
        qdrant_cfg = get_qdrant_config()
        for method in AVAILABLE_METHODS:
            if qdrant.collection_exists(qdrant_cfg.collection_name(method)):
                dostepne_vec.append(method)
        qdrant.close()
        sources["vector"] = bool(dostepne_vec)
    except Exception:
        pass

    try:
        from src.graph_rag.graph_store import Neo4jStore
        graph = Neo4jStore()
        stats = graph.get_stats()
        graph.close()
        sources["graph"] = stats.get("entities", 0) > 0
    except Exception:
        pass

    trees = st.session_state.get("pageindex_trees", {})
    sources["pageindex"] = len(trees) > 0

    # Status źródeł + memory
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔷 Vector", "✅" if sources["vector"] else "❌")
    col2.metric("🕸️ Graph", "✅" if sources["graph"] else "❌")
    col3.metric("🌳 PageIndex", "✅" if sources["pageindex"] else "❌")

    mem_stats = memory.get_stats()
    col4.metric("💭 Memory", f"{mem_stats['short_term_entries']} sesja")

    if not any(sources.values()):
        st.warning("Żadne źródło nie jest dostępne. Zaindeksuj artykuły / zbuduj graf / zbuduj drzewa PageIndex.")
        return

    # --- Mode selection ---
    st.markdown("---")
    mode = st.radio(
        "Tryb",
        ["🤖 Auto (orkiestrator decyduje)", "🎛️ Manual (ty wybierasz)"],
        horizontal=True,
        key="ce_mode",
    )

    is_auto = mode.startswith("🤖")

    # --- Konfiguracja ---
    st.markdown("**Query Augmentation:**")
    col_aug1, col_aug2 = st.columns(2)
    with col_aug1:
        do_rewrite = st.checkbox("Query rewriting", value=True, key="ce_rewrite")
    with col_aug2:
        do_fanout = st.checkbox("Query fan-out", value=True, key="ce_fanout")

    if not is_auto:
        st.markdown("**Wybór źródeł (manual):**")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            use_vector = st.checkbox("Vector", value=sources["vector"], disabled=not sources["vector"], key="ce_vec")
        with col_s2:
            use_graph = st.checkbox("Graph", value=sources["graph"], disabled=not sources["graph"], key="ce_graph")
        with col_s3:
            use_pageindex = st.checkbox("PageIndex", value=sources["pageindex"], disabled=not sources["pageindex"], key="ce_pi")

    col_k, col_b = st.columns(2)
    with col_k:
        top_k = st.number_input("Top-K per źródło", value=5, min_value=1, max_value=15, key="ce_k")
    with col_b:
        total_budget = st.number_input("Budget tokenów", value=100000, min_value=10000, max_value=200000, step=10000, key="ce_budget")

    use_memory = st.checkbox("Dołącz pamięć sesji do kontekstu", value=True, key="ce_use_mem")

    if sources["vector"] and dostepne_vec:
        vec_method = st.selectbox("Kolekcja Qdrant", dostepne_vec, key="ce_vec_method")
    else:
        vec_method = None

    if sources["pageindex"]:
        pi_keys = list(trees.keys())
        pi_key = st.selectbox("Drzewo PageIndex", pi_keys, key="ce_pi_key") if pi_keys else None
    else:
        pi_key = None

    query = st.text_input("Pytanie", key="ce_query")

    if st.button("🎯 Uruchom Context Engineering", type="primary", disabled=not query):
        from src.context_eng.query_augmenter import QueryAugmenter
        from src.context_eng.orchestrator import Orchestrator, OrchestratorDecision
        from src.context_eng.context_assembler import ContextAssembler, ContextBudget
        from src.context_eng.memory import MemoryEntry
        from src.generation.claude_gen import ClaudeGenerator

        # --- 1. Query Augmentation ---
        st.markdown("---")
        st.subheader("Krok 1: Query Augmentation")

        augmenter = QueryAugmenter()
        with st.spinner("Augmentacja zapytania..."):
            augmented = augmenter.augment(query, do_rewrite=do_rewrite, do_fanout=do_fanout)
        augmenter.close()

        col_i1, col_i2 = st.columns([1, 3])
        col_i1.markdown(f"**Intencja:** `{augmented.intent}`")
        col_i2.caption(augmented.reasoning)

        if augmented.rewritten != query:
            st.markdown(f"**Przeformułowane:** _{augmented.rewritten}_")

        if len(augmented.sub_queries) > 1:
            with st.expander(f"Sub-zapytania ({len(augmented.sub_queries)})"):
                for i, sq in enumerate(augmented.sub_queries, 1):
                    st.markdown(f"  {i}. {sq}")

        # --- 2. Orchestrator decision ---
        st.markdown("---")
        st.subheader("Krok 2: Orkiestrator")

        if is_auto:
            orchestrator = Orchestrator()
            with st.spinner("Orkiestrator decyduje..."):
                decision = orchestrator.decide(
                    query=query,
                    available_sources=sources,
                    intent=augmented.intent,
                )
            orchestrator.close()

            st.info(f"**Decyzja:** {decision.reasoning}")
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Vector", "✅ ON" if decision.use_vector else "⬜ OFF")
            col_d2.metric("Graph", "✅ ON" if decision.use_graph else "⬜ OFF")
            col_d3.metric("PageIndex", "✅ ON" if decision.use_pageindex else "⬜ OFF")

            final_use_vector = decision.use_vector
            final_use_graph = decision.use_graph
            final_use_pageindex = decision.use_pageindex
        else:
            st.info("Manual mode — używam twoich ustawień.")
            final_use_vector = use_vector
            final_use_graph = use_graph
            final_use_pageindex = use_pageindex

        # --- 3. Multi-source retrieval ---
        st.markdown("---")
        st.subheader("Krok 3: Wieloźródłowy retrieval")

        vector_results = None
        graph_contexts = None
        pageindex_nodes = None

        # Użyj rewritten query dla retrieval (lepsze dopasowanie)
        retrieval_query = augmented.rewritten

        if final_use_vector and vec_method:
            from src.retrieval.hybrid_search import HybridRetriever
            from config.settings import get_qdrant_config
            qdrant_cfg = get_qdrant_config()
            collection_name = qdrant_cfg.collection_name(vec_method)
            with st.spinner("Vector retrieval..."):
                retriever = HybridRetriever()
                vector_results = retriever.search(retrieval_query, collection_name, limit=top_k)
                retriever.store.close()
            st.caption(f"✅ Vector: {len(vector_results)} chunków")

        if final_use_graph:
            from src.graph_rag.graph_retriever import GraphRetriever
            with st.spinner("Graph retrieval..."):
                graph_ret = GraphRetriever()
                graph_result = graph_ret.search(retrieval_query, limit_per_entity=top_k)
                graph_contexts = graph_result.graph_contexts
                graph_ret.close()
            st.caption(f"✅ Graph: {len(graph_contexts)} encji, łącznie {sum(len(c.mentions) for c in graph_contexts)} wzmianek")

        if final_use_pageindex and pi_key:
            from src.pageindex.navigator import TreeNavigator
            with st.spinner("PageIndex navigation..."):
                tree = trees[pi_key]
                nav = TreeNavigator()
                pi_result = nav.navigate(tree, retrieval_query)
                pageindex_nodes = pi_result.selected_nodes
                nav.close()
            st.caption(f"✅ PageIndex: {len(pageindex_nodes)} sekcji")

        # --- 4. Context Assembly ---
        st.markdown("---")
        st.subheader("Krok 4: Context Assembly")

        # Rezerwuj tokeny dla memory
        memory_budget = 3000 if use_memory else 0
        budget = ContextBudget(
            total=total_budget,
            history=5000 + memory_budget,
        )

        assembler = ContextAssembler(budget=budget)

        with st.spinner("Assembling..."):
            assembled = assembler.assemble(
                query=query,
                vector_results=vector_results,
                graph_contexts=graph_contexts,
                pageindex_nodes=pageindex_nodes,
            )

        # Faceted metadata
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Fragmenty", assembled.facets.total_found)
        col_m2.metric("Źródła", len(assembled.sources_used))
        col_m3.metric("Tokeny (est.)", f"{assembled.total_tokens_estimate:,}")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown("**Per źródło (faceted):**")
            for src, count in assembled.facets.by_source.items():
                st.text(f"  {src}: {count}")
        with col_f2:
            st.markdown("**Per artykuł:**")
            for slug, count in list(assembled.facets.by_article.items())[:5]:
                st.text(f"  {slug[:25]}: {count}")
        with col_f3:
            st.markdown("**Per sekcja:**")
            for sec, count in list(assembled.facets.by_section.items())[:5]:
                st.text(f"  {sec[:25]}: {count}")

        # Budget progress
        progress_val = min(assembled.total_tokens_estimate / budget.available_for_retrieval(), 1.0)
        st.progress(progress_val, text=f"{assembled.total_tokens_estimate:,} / {budget.available_for_retrieval():,} tokenów retrieval")

        # Budget pie + Sankey flow
        col_pie, col_sankey = st.columns(2)

        with col_pie:
            import plotly.graph_objects as go
            fig_pie = go.Figure(data=[go.Pie(
                labels=["System prompt", "History", "Retrieval", "Response reserve"],
                values=[
                    budget.system_prompt,
                    budget.history,
                    assembled.total_tokens_estimate,
                    budget.response_reserve,
                ],
                hole=0.4,
            )])
            fig_pie.update_layout(title="Podział budżetu tokenów", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_sankey:
            # Sankey: źródła → assembler → LLM
            labels = []
            sources = []
            targets = []
            values = []

            src_idx = {}
            for src in assembled.facets.by_source:
                labels.append(f"{src}\n({assembled.facets.by_source[src]})")
                src_idx[src] = len(labels) - 1

            assembler_idx = len(labels)
            labels.append("Context\nAssembler")

            llm_idx = len(labels)
            labels.append("LLM\n(Claude)")

            for src, count in assembled.facets.by_source.items():
                sources.append(src_idx[src])
                targets.append(assembler_idx)
                values.append(count)

            sources.append(assembler_idx)
            targets.append(llm_idx)
            values.append(len(assembled.ranked_contexts))

            if sources:
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15, thickness=20,
                        label=labels,
                        color=["#4090e0", "#ffaa00", "#44bb44", "#888", "#222"][:len(labels)],
                    ),
                    link=dict(source=sources, target=targets, value=values),
                )])
                fig_sankey.update_layout(title="Przepływ kontekstu", height=350)
                st.plotly_chart(fig_sankey, use_container_width=True)

        # --- 5. Memory integration ---
        memory_context = ""
        if use_memory:
            st.markdown("---")
            st.subheader("Krok 5: Memory")
            memory_context = memory.get_session_context(max_entries=3)
            if memory_context:
                with st.expander(f"Kontekst sesji ({len(memory.get_short_term())} poprzednich interakcji)"):
                    st.text(memory_context)
            else:
                st.caption("Brak historii sesji — to pierwsze zapytanie.")

        # --- 6. Generation ---
        st.markdown("---")
        st.subheader("Krok 6: Odpowiedź")

        generator = ClaudeGenerator()

        # Dołącz memory do kontekstu
        final_contexts = assembled.ranked_contexts.copy()
        if memory_context:
            final_contexts.insert(0, memory_context)

        with st.spinner("Generowanie odpowiedzi z hybrydowego kontekstu..."):
            answer = generator.generate(query, final_contexts)

        generator.close()

        st.markdown(answer)

        # --- 7. Zapisz do memory ---
        entry = MemoryEntry(
            query=query,
            answer=answer,
            timestamp=_timestamp(),
            intent=augmented.intent,
            sources_used=assembled.sources_used,
            num_contexts=len(assembled.ranked_contexts),
        )
        memory.add(entry)
        st.caption(f"💾 Zapisano do memory (sesja: {memory._session_id})")

        # Pokaż konteksty
        with st.expander(f"Użyty kontekst ({len(final_contexts)} fragmentów)"):
            for i, ctx in enumerate(final_contexts, 1):
                st.markdown(f"**[{i}]**")
                st.text(ctx[:400] + ("..." if len(ctx) > 400 else ""))
                st.markdown("---")

    # --- Memory management ---
    st.markdown("---")
    with st.expander("Memory management"):
        mem_stats = memory.get_stats()
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Short-term (sesja)", mem_stats["short_term_entries"])
        col_m2.metric("Episodic sesje", mem_stats["episodic_sessions"])
        col_m3.metric("Episodic łącznie", mem_stats["episodic_total_entries"])

        # Historia sesji
        if memory.get_short_term():
            st.markdown("**Historia bieżącej sesji:**")
            for i, entry in enumerate(memory.get_short_term(), 1):
                st.text(f"  [{i}] {entry.intent} | {entry.query[:80]}")

        # Episodic search
        search_term = st.text_input("Szukaj w episodic memory", key="mem_search")
        if search_term:
            results = memory.search_episodic(search_term, limit=5)
            if results:
                for r in results:
                    st.text(f"  [{r.timestamp}] {r.query}")
                    st.caption(f"    → {r.answer[:150]}...")
            else:
                st.caption("Brak wyników.")

        if st.button("🗑️ Wyczyść short-term"):
            memory.clear_short_term()
            st.rerun()


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
