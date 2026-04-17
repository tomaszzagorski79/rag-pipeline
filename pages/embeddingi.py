"""Strona embeddingów tytułów — wektory tytułów artykułów do Qdrant."""

import streamlit as st
from pathlib import Path
import sys
import time
import yaml

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

COLLECTION_NAME = "articles_titles"


def _wczytaj_artykuly_z_raw(raw_dir: Path) -> list[dict]:
    """Wczytaj tytuły i metadane z pobranych plików markdown."""
    artykuly = []
    if not raw_dir.exists():
        return artykuly

    for plik in sorted(raw_dir.glob("*.md")):
        tekst = plik.read_text(encoding="utf-8")
        if tekst.startswith("---"):
            parts = tekst.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1])
                    artykuly.append({
                        "title": meta.get("title", plik.stem),
                        "source_url": meta.get("source_url", ""),
                        "slug": plik.stem,
                        "source": "pobrany",
                    })
                except Exception:
                    artykuly.append({
                        "title": plik.stem,
                        "source_url": "",
                        "slug": plik.stem,
                        "source": "pobrany",
                    })
    return artykuly


def _wczytaj_urls_z_listy(urls_file: Path, pobrane_urls: set) -> list[dict]:
    """Wczytaj URL-e z article_urls.txt które nie zostały jeszcze pobrane."""
    if not urls_file.exists():
        return []

    import re
    urls = []
    for linia in urls_file.read_text(encoding="utf-8").splitlines():
        linia = linia.strip()
        if linia and not linia.startswith("#") and linia not in pobrane_urls:
            slug = linia.rstrip("/").split("/")[-1]
            slug = re.sub(r"[^\w\-]", "_", slug)[:100]
            urls.append({
                "title": None,  # tytuł do pobrania
                "source_url": linia,
                "slug": slug,
                "source": "url",
            })
    return urls


def _pobierz_tytul_jina(url: str) -> str:
    """Pobierz tytuł strony przez Jina Reader (tylko nagłówek, nie cała treść)."""
    import httpx
    from config.settings import get_jina_config

    config = get_jina_config()
    reader_url = f"{config.reader_url}{url}"

    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            reader_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "X-Return-Format": "markdown",
            },
        )
        resp.raise_for_status()

    # Wyciągnij pierwszy H1
    for line in resp.text.split("\n"):
        if line.startswith("# "):
            return line.lstrip("# ").strip()

    # Fallback: pierwsza niepusta linia
    for line in resp.text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("[") and not stripped.startswith("!"):
            return stripped[:150]

    return url.rstrip("/").split("/")[-1]


def render():
    st.title("2. Embeddingi tytułów")
    st.markdown(
        "Zamień tytuły artykułów na wektory i wrzuć do Qdrant. "
        "Opcjonalny krok — możesz go pominąć i przejść do chunkingu."
    )
    st.caption("Wizualizacja: otwórz kolekcję w Qdrant Cloud dashboard → Visualize")

    raw_dir = _root / "data" / "raw"
    urls_file = _root / "data" / "article_urls.txt"

    # --- Załaduj źródła ---
    artykuly_raw = _wczytaj_artykuly_z_raw(raw_dir)
    pobrane_urls = {a["source_url"] for a in artykuly_raw}
    artykuly_url = _wczytaj_urls_z_listy(urls_file, pobrane_urls)

    total = len(artykuly_raw) + len(artykuly_url)

    if total == 0:
        st.warning("Brak artykułów. Dodaj URL-e w zakładce Scraping (ręcznie lub z sitemapy).")
        return

    # --- Status kolekcji ---
    try:
        from src.vectorstore.qdrant_store import QdrantStore
        store = QdrantStore()
        if store.collection_exists(COLLECTION_NAME):
            count = store.count_points(COLLECTION_NAME)
            st.metric("Tytułów w Qdrant", count)
        else:
            st.metric("Tytułów w Qdrant", "kolekcja nie istnieje")
        store.close()
    except Exception as e:
        st.warning(f"Nie można połączyć z Qdrant: {e}")

    # --- Pobierz tytuły z URL-i (Jina Reader) ---
    if artykuly_url:
        st.markdown("---")
        st.header(f"Pobierz tytuły ({len(artykuly_url)} URL-i bez tytułu)")
        st.caption("URL-e z listy które nie zostały jeszcze pobrane — tytuł zostanie pobrany przez Jina Reader.")

        # Cache w session_state
        if "fetched_titles" not in st.session_state:
            st.session_state["fetched_titles"] = {}

        fetched = st.session_state["fetched_titles"]
        nie_pobrane = [a for a in artykuly_url if a["source_url"] not in fetched]

        if nie_pobrane:
            if st.button(f"📥 Pobierz tytuły ({len(nie_pobrane)} URL-i)", type="secondary"):
                progress = st.progress(0, text="Pobieranie tytułów...")

                for i, art in enumerate(nie_pobrane):
                    progress.progress(
                        (i + 1) / len(nie_pobrane),
                        text=f"Pobieranie tytułu {i+1}/{len(nie_pobrane)}...",
                    )
                    try:
                        tytul = _pobierz_tytul_jina(art["source_url"])
                        fetched[art["source_url"]] = tytul
                    except Exception as e:
                        fetched[art["source_url"]] = art["slug"]

                    if i < len(nie_pobrane) - 1:
                        time.sleep(1)

                progress.progress(1.0, text="Gotowe!")
                st.session_state["fetched_titles"] = fetched
                st.rerun()

        # Uzupełnij tytuły z cache
        for art in artykuly_url:
            if art["source_url"] in fetched:
                art["title"] = fetched[art["source_url"]]

    # --- Połącz wszystkie artykuły ---
    wszystkie = artykuly_raw + [a for a in artykuly_url if a["title"]]
    artykuly_bez_tytulu = [a for a in artykuly_url if not a["title"]]

    if artykuly_bez_tytulu:
        st.info(
            f"{len(artykuly_bez_tytulu)} URL-i bez tytułu — "
            f"kliknij 'Pobierz tytuły' powyżej."
        )

    if not wszystkie:
        st.warning("Brak artykułów z tytułami do embeddingu.")
        return

    # --- Wybór tytułów ---
    st.markdown("---")
    st.header("Wybierz tytuły do embeddingu")

    col_all, col_info = st.columns([1, 5])
    with col_all:
        select_all = st.checkbox("Zaznacz wszystkie", value=True)
    with col_info:
        st.caption(f"Dostępnych: {len(wszystkie)} ({len(artykuly_raw)} pobranych + {len(wszystkie) - len(artykuly_raw)} z URL-i)")

    selected = []
    for art in wszystkie:
        label = art["title"]
        if art["source"] == "url":
            label += " (z URL)"

        checked = st.checkbox(
            label,
            value=select_all,
            key=f"title_{art['slug']}",
            help=art["source_url"],
        )
        if checked:
            selected.append(art)

    st.caption(f"Zaznaczono: {len(selected)} / {len(wszystkie)}")

    # --- Embed i wyślij ---
    col_embed, col_clear = st.columns(2)

    with col_embed:
        if st.button("🚀 Embed tytuły → Qdrant", type="primary", disabled=len(selected) == 0):
            from src.embeddings.jina_embed import JinaDenseEmbedder
            from src.vectorstore.qdrant_store import QdrantStore

            with st.spinner(f"Generowanie embeddingów {len(selected)} tytułów..."):
                embedder = JinaDenseEmbedder()
                store = QdrantStore()

                titles = [a["title"] for a in selected]
                metadata = [
                    {"source_url": a["source_url"], "slug": a["slug"]}
                    for a in selected
                ]

                vectors = embedder.embed_documents(titles)

                store.create_dense_only_collection(COLLECTION_NAME)
                store.upsert_titles(COLLECTION_NAME, titles, vectors, metadata)

                count = store.count_points(COLLECTION_NAME)

                embedder.close()
                store.close()

            st.success(f"Zaindeksowano {len(selected)} tytułów. Łącznie w kolekcji: {count}")
            st.rerun()

    with col_clear:
        if st.button("🗑️ Wyczyść kolekcję"):
            try:
                from src.vectorstore.qdrant_store import QdrantStore
                store = QdrantStore()
                store.delete_collection(COLLECTION_NAME)
                store.close()
                st.success("Kolekcja usunięta.")
                st.rerun()
            except Exception as e:
                st.error(f"Błąd: {e}")

    # --- Wizualizacja UMAP / t-SNE ---
    st.markdown("---")
    st.header("📊 Wizualizacja 2D (UMAP / t-SNE)")
    st.caption("Projekcja wektorów 1024d → 2D. Tytuły blisko siebie = podobne tematycznie.")

    try:
        from src.vectorstore.qdrant_store import QdrantStore
        store = QdrantStore()
        if not store.collection_exists(COLLECTION_NAME):
            st.info("Kolekcja tytułów nie istnieje — zaindeksuj tytuły powyżej.")
            store.close()
            return
        count = store.count_points(COLLECTION_NAME)
        store.close()
    except Exception as e:
        st.warning(f"Qdrant: {e}")
        return

    if count < 3:
        st.info(f"Za mało tytułów ({count}) do wizualizacji — potrzeba min. 3.")
        return

    col_method, col_btn = st.columns([2, 1])
    with col_method:
        viz_method = st.radio(
            "Algorytm redukcji wymiarów",
            ["UMAP", "t-SNE"],
            horizontal=True,
            key="viz_method",
            help="UMAP: zachowuje lokalną + globalną strukturę. t-SNE: tylko lokalna (klastry).",
        )

    with col_btn:
        run_viz = st.button("🎨 Generuj wizualizację", type="primary")

    if run_viz:
        import numpy as np
        import plotly.express as px
        from qdrant_client import models

        from src.vectorstore.qdrant_store import QdrantStore

        with st.spinner(f"Pobieranie {count} wektorów z Qdrant..."):
            store = QdrantStore()
            scroll_result, _ = store.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                with_payload=True,
                with_vectors=True,
            )
            store.close()

        vectors = np.array([p.vector for p in scroll_result])
        titles = [(p.payload or {}).get("title", "?") for p in scroll_result]
        slugs = [(p.payload or {}).get("slug", "?") for p in scroll_result]

        with st.spinner(f"{viz_method} redukcja 1024d → 2D..."):
            if viz_method == "UMAP":
                import umap
                n_neighbors = min(15, max(2, len(vectors) - 1))
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
                coords = reducer.fit_transform(vectors)
            else:
                from sklearn.manifold import TSNE
                perplexity = min(30, max(2, len(vectors) - 1))
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                coords = reducer.fit_transform(vectors)

        # Plotly scatter
        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            hover_name=titles,
            hover_data={"slug": slugs},
            labels={"x": f"{viz_method} 1", "y": f"{viz_method} 2"},
            title=f"Projekcja {len(vectors)} tytułów w 2D ({viz_method})",
        )
        fig.update_traces(marker=dict(size=12, opacity=0.8))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "💡 **Interpretacja:** punkty blisko siebie = tytuły semantycznie podobne. "
            "Klastry = grupy tematyczne. Outliers = wyjątki w korpusie."
        )
