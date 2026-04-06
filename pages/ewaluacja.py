"""Strona ewaluacji RAGAS — dashboard z wynikami i wykresami."""

import json
import streamlit as st
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("4. Ewaluacja RAGAS")
    st.markdown("Mierzenie jakości pipeline'u RAG: faithfulness, relevancy, precision.")

    results_dir = _root / "data" / "results"
    test_set_file = _root / "data" / "test_set.json"

    # --- Zarządzanie zestawem testowym ---
    st.header("Zestaw testowy")

    current_test_set = "[]"
    if test_set_file.exists():
        current_test_set = test_set_file.read_text(encoding="utf-8")

    try:
        test_data = json.loads(current_test_set)
    except Exception:
        test_data = []

    st.metric("Pytań testowych", len(test_data))

    with st.expander("Edytuj zestaw testowy (JSON)", expanded=len(test_data) <= 1):
        st.caption(
            "Format: lista obiektów z kluczami: question, ground_truth, source_article"
        )

        edited = st.text_area(
            "test_set.json",
            value=json.dumps(test_data, ensure_ascii=False, indent=2),
            height=400,
            key="test_set_editor",
        )

        if st.button("💾 Zapisz zestaw testowy"):
            try:
                parsed = json.loads(edited)
                test_set_file.write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.success(f"Zapisano {len(parsed)} pytań testowych.")
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Nieprawidłowy JSON: {e}")

    # Lista aktualnych pytań
    if test_data:
        with st.expander(f"Aktualne pytania ({len(test_data)})", expanded=True):
            for i, q in enumerate(test_data):
                col_q, col_del = st.columns([9, 1])
                with col_q:
                    st.markdown(f"**{i+1}.** {q['question']}")
                    st.caption(f"Ground truth: {q['ground_truth'][:100]}...")
                with col_del:
                    if st.button("🗑️", key=f"del_{i}"):
                        test_data.pop(i)
                        test_set_file.write_text(
                            json.dumps(test_data, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        st.rerun()

    # Dodaj nowe pytanie
    with st.expander("Dodaj nowe pytanie"):
        new_q = st.text_input("Pytanie", key="new_question")
        new_gt = st.text_area("Oczekiwana odpowiedź (ground truth)", key="new_gt")
        new_src = st.text_input("Artykuł źródłowy (opcjonalnie)", key="new_src")

        if st.button("➕ Dodaj pytanie") and new_q and new_gt:
            test_data.append({
                "question": new_q,
                "ground_truth": new_gt,
                "source_article": new_src,
            })
            test_set_file.write_text(
                json.dumps(test_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.success("Dodano pytanie.")
            st.rerun()

    # --- Uruchom ewaluację ---
    st.markdown("---")
    st.header("Uruchom ewaluację")

    # Sprawdź dostępne kolekcje
    dostepne = []
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()
        for method in ["naive", "header", "semantic"]:
            name = qdrant_cfg.collection_name(method)
            if store.collection_exists(name):
                dostepne.append(method)
        store.close()
    except Exception:
        pass

    if not dostepne:
        st.warning("Brak kolekcji w Qdrant. Najpierw zaindeksuj artykuły.")
    elif len(test_data) < 1:
        st.warning("Dodaj przynajmniej 1 pytanie testowe.")
    else:
        eval_methods = st.multiselect(
            "Metody do ewaluacji",
            dostepne,
            default=dostepne,
            key="eval_methods",
        )
        eval_limit = st.number_input("Top-K kontekstów", value=5, min_value=1, max_value=20, key="eval_k")

        if st.button("🧪 Uruchom ewaluację RAGAS", type="primary"):
            with st.spinner("Ewaluacja w toku... (to może potrwać kilka minut)"):
                try:
                    from src.evaluation.ragas_eval import run_evaluation, zapisz_wyniki

                    wyniki = run_evaluation(eval_methods, limit=eval_limit)
                    if wyniki:
                        plik = zapisz_wyniki(wyniki)
                        st.success(f"Ewaluacja zakończona! Wyniki w: {plik.name}")
                        st.rerun()
                    else:
                        st.error("Brak wyników ewaluacji.")
                except Exception as e:
                    st.error(f"Błąd ewaluacji: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # --- Dashboard wyników ---
    st.markdown("---")
    st.header("Wyniki ewaluacji")

    result_files = sorted(results_dir.glob("evaluation_*.json"), reverse=True) if results_dir.exists() else []

    if not result_files:
        st.info("Brak wyników ewaluacji. Uruchom ewaluację powyżej.")
        return

    selected_file = st.selectbox(
        "Wybierz ewaluację",
        result_files,
        format_func=lambda p: p.name,
    )

    if selected_file:
        data = json.loads(selected_file.read_text(encoding="utf-8"))

        # --- Tabela porównawcza ---
        st.subheader("Porównanie metod")

        import pandas as pd

        rows = []
        for method, method_data in data.items():
            scores = method_data.get("scores", {})
            if scores:
                row = {"Metoda": method}
                row.update({k: round(v, 4) for k, v in scores.items()})
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows).set_index("Metoda")
            st.dataframe(df.style.highlight_max(axis=0, color="#90EE90"), use_container_width=True)

            # --- Wykres radarowy ---
            st.subheader("Wykres porównawczy")

            import plotly.graph_objects as go

            metrics = [c for c in df.columns]

            fig = go.Figure()
            for method in df.index:
                values = df.loc[method].tolist()
                values.append(values[0])  # zamknij radar
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill="toself",
                    name=method,
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Wykres słupkowy ---
            st.subheader("Porównanie słupkowe")

            import plotly.express as px

            df_melted = df.reset_index().melt(id_vars="Metoda", var_name="Metryka", value_name="Wynik")
            fig_bar = px.bar(
                df_melted,
                x="Metryka",
                y="Wynik",
                color="Metoda",
                barmode="group",
                range_y=[0, 1],
                height=400,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Szczegóły per pytanie ---
        st.subheader("Wyniki per pytanie")

        for method, method_data in data.items():
            per_q = method_data.get("per_question", [])
            if per_q:
                with st.expander(f"Szczegóły: {method}"):
                    df_q = pd.DataFrame(per_q)
                    # Kolumny do wyświetlenia
                    display_cols = [c for c in df_q.columns if c != "response" and c != "retrieved_contexts"]
                    if display_cols:
                        st.dataframe(df_q[display_cols], use_container_width=True, hide_index=True)

            if method_data.get("error"):
                st.error(f"Błąd dla {method}: {method_data['error']}")
