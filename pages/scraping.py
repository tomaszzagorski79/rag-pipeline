"""Strona scrapingu artykułów."""

import streamlit as st
from pathlib import Path
import sys

# Dodaj root do path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


def render():
    st.title("1. Scraping artykułów")
    st.markdown("Pobieranie artykułów ze stron WWW przez Jina Reader API.")

    data_dir = _root / "data"
    raw_dir = data_dir / "raw"
    urls_file = data_dir / "article_urls.txt"

    # --- Import z sitemapy ---
    st.header("Import z sitemapy")
    st.markdown("Pobierz URL-e artykułów z pliku sitemap.xml.")

    col_sitemap_url, col_sitemap_file = st.columns(2)

    with col_sitemap_url:
        sitemap_url = st.text_input(
            "URL sitemapy",
            placeholder="https://example.com/sitemap.xml",
        )
        if st.button("📥 Pobierz z URL") and sitemap_url:
            try:
                import httpx
                from xml.etree import ElementTree

                resp = httpx.get(sitemap_url, timeout=15.0, follow_redirects=True)
                resp.raise_for_status()
                root = ElementTree.fromstring(resp.text)

                ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                locs = [loc.text for loc in root.findall(".//s:loc", ns) if loc.text]

                if not locs:
                    locs = [loc.text for loc in root.findall(".//loc") if loc.text]

                locs = locs[:300]
                st.session_state["sitemap_urls"] = locs
                st.success(f"Znaleziono {len(locs)} URL-i")
            except Exception as e:
                st.error(f"Błąd: {e}")

    with col_sitemap_file:
        uploaded = st.file_uploader("Lub wgraj plik sitemap.xml", type=["xml"])
        if uploaded:
            try:
                from xml.etree import ElementTree

                root = ElementTree.fromstring(uploaded.read())
                ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                locs = [loc.text for loc in root.findall(".//s:loc", ns) if loc.text]

                if not locs:
                    locs = [loc.text for loc in root.findall(".//loc") if loc.text]

                locs = locs[:300]
                st.session_state["sitemap_urls"] = locs
                st.success(f"Znaleziono {len(locs)} URL-i")
            except Exception as e:
                st.error(f"Błąd parsowania XML: {e}")

    if "sitemap_urls" in st.session_state and st.session_state["sitemap_urls"]:
        sitemap_locs = st.session_state["sitemap_urls"]
        with st.expander(f"Znalezione URL-e ({len(sitemap_locs)})", expanded=False):
            for u in sitemap_locs[:50]:
                st.text(u)
            if len(sitemap_locs) > 50:
                st.caption(f"... i {len(sitemap_locs) - 50} więcej")

        if st.button("➕ Dodaj do listy URL-i"):
            existing_text = urls_file.read_text(encoding="utf-8") if urls_file.exists() else ""
            existing_set = {
                l.strip() for l in existing_text.splitlines()
                if l.strip() and not l.strip().startswith("#")
            }
            nowe = [u for u in sitemap_locs if u not in existing_set]

            if nowe:
                dodane = "\n".join(nowe)
                urls_file.write_text(
                    existing_text.rstrip() + "\n" + dodane + "\n",
                    encoding="utf-8",
                )
                st.success(f"Dodano {len(nowe)} nowych URL-i (pominięto {len(sitemap_locs) - len(nowe)} duplikatów)")
                del st.session_state["sitemap_urls"]
                st.rerun()
            else:
                st.info("Wszystkie URL-e już są na liście.")

    # --- Zarządzanie URL-ami ---
    st.markdown("---")
    st.header("Lista URL-i")

    current_urls = ""
    if urls_file.exists():
        current_urls = urls_file.read_text(encoding="utf-8")

    new_urls = st.text_area(
        "Wklej URL-e artykułów (jeden na linię, # = komentarz)",
        value=current_urls,
        height=300,
        help="Zapisz listę URL-i klikając przycisk poniżej.",
    )

    if st.button("💾 Zapisz listę URL-i", type="primary"):
        urls_file.write_text(new_urls, encoding="utf-8")
        st.success("Lista URL-i zapisana!")
        st.rerun()

    # Policz aktywne URL-e
    active_urls = [
        l.strip()
        for l in new_urls.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    st.caption(f"Aktywnych URL-i: {len(active_urls)}")

    # --- Scraping ---
    st.markdown("---")
    st.header("Pobieranie artykułów")

    col1, col2 = st.columns(2)

    with col1:
        # Status pobranych
        existing = list(raw_dir.glob("*.md")) if raw_dir.exists() else []
        st.metric("Pobrane artykuły", len(existing))

        if existing:
            with st.expander("Pokaż pobrane pliki"):
                for f in sorted(existing):
                    size = f.stat().st_size
                    st.text(f"  {f.name} ({size:,} B)")

    with col2:
        # Pobierz pojedynczy URL
        single_url = st.text_input("Pobierz pojedynczy URL", placeholder="https://...")
        if st.button("📥 Pobierz artykuł") and single_url:
            with st.spinner("Pobieranie..."):
                try:
                    from src.scraper.jina_reader import pobierz_artykul, zapisz_artykul
                    artykul = pobierz_artykul(single_url)
                    plik = zapisz_artykul(artykul)
                    st.success(f"Zapisano: {plik.name} ({len(artykul['markdown'])} znaków)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Błąd: {e}")

    # Pobierz wszystkie
    st.markdown("---")
    if st.button("🚀 Pobierz wszystkie artykuły z listy", type="primary", disabled=len(active_urls) == 0):
        from src.scraper.jina_reader import pobierz_artykul, zapisz_artykul
        import time

        progress = st.progress(0, text="Rozpoczynanie...")
        log = st.empty()

        pobrane = 0
        bledy = 0

        for i, url in enumerate(active_urls):
            progress.progress(
                (i + 1) / len(active_urls),
                text=f"Pobieranie {i+1}/{len(active_urls)}...",
            )
            log.text(f"→ {url}")

            try:
                artykul = pobierz_artykul(url)
                zapisz_artykul(artykul)
                pobrane += 1
            except Exception as e:
                bledy += 1
                st.warning(f"Błąd dla {url}: {e}")

            # Rate limiting
            if i < len(active_urls) - 1:
                time.sleep(3)

        progress.progress(1.0, text="Zakończono!")
        st.success(f"Pobrano {pobrane}/{len(active_urls)} artykułów. Błędy: {bledy}")
        st.rerun()

    # --- Podgląd artykułu ---
    st.markdown("---")
    st.header("Podgląd artykułu")

    existing = sorted(raw_dir.glob("*.md")) if raw_dir.exists() else []
    if existing:
        selected = st.selectbox(
            "Wybierz artykuł",
            existing,
            format_func=lambda p: p.name,
        )
        if selected:
            content = selected.read_text(encoding="utf-8")

            # Pokaż metadane
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    st.code(parts[1].strip(), language="yaml")
                    st.markdown(parts[2][:3000] + ("..." if len(parts[2]) > 3000 else ""))
                    st.caption(f"Długość treści: {len(parts[2])} znaków")
            else:
                st.markdown(content[:3000])
    else:
        st.info("Brak pobranych artykułów. Pobierz je powyżej.")
