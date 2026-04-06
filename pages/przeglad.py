"""Strona przeglądu — status pipeline'u i szybki start."""

import streamlit as st
from pathlib import Path


def render():
    st.title("RAG Pipeline — Przegląd")
    st.markdown("Lokalne środowisko RAG z hybrid search i ewaluacją RAGAS.")

    # Status pipeline'u
    st.header("Status pipeline'u")

    data_dir = Path(__file__).resolve().parent.parent / "data"
    raw_dir = data_dir / "raw"
    results_dir = data_dir / "results"
    urls_file = data_dir / "article_urls.txt"
    test_set = data_dir / "test_set.json"

    # Liczniki
    n_urls = 0
    if urls_file.exists():
        lines = urls_file.read_text(encoding="utf-8").splitlines()
        n_urls = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

    n_articles = len(list(raw_dir.glob("*.md"))) if raw_dir.exists() else 0

    n_test_questions = 0
    if test_set.exists():
        import json
        try:
            data = json.loads(test_set.read_text(encoding="utf-8"))
            n_test_questions = len(data)
        except Exception:
            pass

    n_results = len(list(results_dir.glob("evaluation_*.json"))) if results_dir.exists() else 0

    # Kolekcje Qdrant
    n_collections = 0
    collection_info = {}
    try:
        from config.settings import get_qdrant_config
        from src.vectorstore.qdrant_store import QdrantStore
        store = QdrantStore()
        qdrant_cfg = get_qdrant_config()
        for method in ["naive", "header", "semantic"]:
            name = qdrant_cfg.collection_name(method)
            if store.collection_exists(name):
                n_collections += 1
                collection_info[method] = store.count_points(name)
        store.close()
    except Exception:
        pass

    # Metryki w kolumnach
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("URL-e do scrapowania", n_urls)
    col2.metric("Zescrapowane artykuły", n_articles)
    col3.metric("Kolekcje Qdrant", n_collections)
    col4.metric("Ewaluacje RAGAS", n_results)

    # Szczegóły kolekcji
    if collection_info:
        st.subheader("Kolekcje w Qdrant Cloud")
        cols = st.columns(3)
        for i, (method, count) in enumerate(collection_info.items()):
            cols[i].metric(f"articles_{method}", f"{count} punktów")

    # Workflow
    st.markdown("---")
    st.header("Workflow")
    steps = [
        ("1. Konfiguracja", "Uzupełnij `.env` z kluczami API", n_urls > 0 or n_articles > 0),
        ("2. Scraping", f"Pobierz artykuły ({n_articles} gotowych)", n_articles > 0),
        ("3. Chunking & Indeksowanie", f"Zaindeksuj w Qdrant ({n_collections}/3 kolekcji)", n_collections > 0),
        ("4. Pytania testowe", f"Przygotuj test_set.json ({n_test_questions} pytań)", n_test_questions > 1),
        ("5. Ewaluacja", f"Uruchom RAGAS ({n_results} ewaluacji)", n_results > 0),
    ]

    for name, desc, done in steps:
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} **{name}** — {desc}")

    # Reset pipeline
    st.markdown("---")
    st.header("Reset pipeline")

    with st.expander("Wyczyść cały pipeline do zera", expanded=False):
        st.warning("Ta operacja usunie WSZYSTKIE dane: artykuły, kolekcje Qdrant, wyniki ewaluacji i pytania testowe.")

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            reset_articles = st.checkbox("Usuń pobrane artykuły (data/raw/)", value=True, key="reset_art")
            reset_urls = st.checkbox("Wyczyść listę URL-i", value=True, key="reset_urls")
            reset_test_set = st.checkbox("Wyczyść pytania testowe", value=True, key="reset_test")
            reset_results = st.checkbox("Usuń wyniki ewaluacji", value=True, key="reset_results")

        with col_r2:
            reset_qdrant = st.checkbox("Usuń kolekcje Qdrant (chunki + tytuły)", value=True, key="reset_qdrant")

        potwierdzenie = st.text_input(
            "Wpisz RESET żeby potwierdzić",
            key="reset_confirm",
        )

        if st.button("🗑️ Resetuj pipeline", type="primary", disabled=potwierdzenie != "RESET"):
            import shutil

            log = []

            if reset_articles and raw_dir.exists():
                for f in raw_dir.glob("*.md"):
                    f.unlink()
                log.append("Usunięto artykuły z data/raw/")

            if reset_urls and urls_file.exists():
                urls_file.write_text(
                    "# Lista URL-i artykułów\n# Jeden URL na linię.\n",
                    encoding="utf-8",
                )
                log.append("Wyczyszczono article_urls.txt")

            if reset_test_set and test_set.exists():
                test_set.write_text("[]", encoding="utf-8")
                log.append("Wyczyszczono test_set.json")

            if reset_results and results_dir.exists():
                for f in results_dir.glob("*.json"):
                    f.unlink()
                for f in results_dir.glob("*.csv"):
                    f.unlink()
                log.append("Usunięto wyniki ewaluacji")

            if reset_qdrant:
                try:
                    from config.settings import get_qdrant_config
                    from src.vectorstore.qdrant_store import QdrantStore

                    store = QdrantStore()
                    qdrant_cfg = get_qdrant_config()

                    for method in ["naive", "header", "semantic"]:
                        name = qdrant_cfg.collection_name(method)
                        if store.collection_exists(name):
                            store.delete_collection(name)
                            log.append(f"Usunięto kolekcję: {name}")

                    if store.collection_exists("articles_titles"):
                        store.delete_collection("articles_titles")
                        log.append("Usunięto kolekcję: articles_titles")

                    store.close()
                except Exception as e:
                    log.append(f"Błąd Qdrant: {e}")

            for msg in log:
                st.write(f"✓ {msg}")

            st.success("Pipeline zresetowany!")
            st.rerun()

    # Szybki start
    st.markdown("---")
    st.header("Szybki start (CLI)")
    st.code(
        """\
# Aktywuj środowisko
.venv\\Scripts\\activate

# 1. Pobierz artykuły
python scripts/01_scrape_articles.py

# 2. Zaindeksuj (wszystkie metody)
python scripts/02_chunk_and_index.py --method all

# 3. Testuj zapytania
python scripts/03_query.py

# 4. Ewaluacja RAGAS
python scripts/04_run_evaluation.py --method all

# 5. Porównanie
python scripts/05_compare_methods.py
""",
        language="bash",
    )
