"""RAG Pipeline — główna aplikacja Streamlit.

Uruchomienie:
    .venv\\Scripts\\streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Nawigacja w sidebarze
st.sidebar.title("RAG Pipeline")
st.sidebar.markdown("---")

strona = st.sidebar.radio(
    "Nawigacja",
    [
        "Przegląd",
        "1. Scraping",
        "2. Chunking & Indeksowanie",
        "3. Zapytania",
        "4. Ewaluacja RAGAS",
    ],
    label_visibility="collapsed",
)

# Status .env
st.sidebar.markdown("---")
st.sidebar.caption("Status konfiguracji")
try:
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent / ".env")
    keys = {
        "QDRANT_URL": bool(os.getenv("QDRANT_URL")),
        "QDRANT_API_KEY": bool(os.getenv("QDRANT_API_KEY")),
        "JINA_API_KEY": bool(os.getenv("JINA_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
    }
    for name, ok in keys.items():
        icon = "✅" if ok else "❌"
        st.sidebar.text(f"{icon} {name}")
except Exception:
    st.sidebar.error("Brak pliku .env")


# --- Routing stron ---

if strona == "Przegląd":
    from pages import przeglad
    przeglad.render()

elif strona == "1. Scraping":
    from pages import scraping
    scraping.render()

elif strona == "2. Chunking & Indeksowanie":
    from pages import chunking
    chunking.render()

elif strona == "3. Zapytania":
    from pages import zapytania
    zapytania.render()

elif strona == "4. Ewaluacja RAGAS":
    from pages import ewaluacja
    ewaluacja.render()
