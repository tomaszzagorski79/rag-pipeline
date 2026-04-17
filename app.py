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
        "2. Embeddingi tytułów",
        "3. Chunking & Indeksowanie",
        "4. Zapytania",
        "5. Re-ranking",
        "6. HyDE",
        "7. Adaptive RAG",
        "8. CRAG",
        "9. Ewaluacja RAGAS",
        "10. Hallucination Detection",
        "11. Benchmarki embeddingów",
        "12. CRQ Scoring",
        "13. PageIndex",
        "14. Agentic RAG",
        "15. Graph RAG",
        "16. Hybrid Vector+Graph",
        "17. Context Engineering",
        "18. RAG-Fusion",
        "19. FLARE",
        "20. RAPTOR",
        "21. Speculative RAG",
        "22. Decision Framework",
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
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
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

elif strona == "2. Embeddingi tytułów":
    from pages import embeddingi
    embeddingi.render()

elif strona == "3. Chunking & Indeksowanie":
    from pages import chunking
    chunking.render()

elif strona == "4. Zapytania":
    from pages import zapytania
    zapytania.render()

elif strona == "5. Re-ranking":
    from pages import reranking
    reranking.render()

elif strona == "6. HyDE":
    from pages import hyde
    hyde.render()

elif strona == "7. Adaptive RAG":
    from pages import adaptive_rag
    adaptive_rag.render()

elif strona == "8. CRAG":
    from pages import crag
    crag.render()

elif strona == "9. Ewaluacja RAGAS":
    from pages import ewaluacja
    ewaluacja.render()

elif strona == "10. Hallucination Detection":
    from pages import hallucination
    hallucination.render()

elif strona == "11. Benchmarki embeddingów":
    from pages import benchmarki
    benchmarki.render()

elif strona == "12. CRQ Scoring":
    from pages import crq_scoring
    crq_scoring.render()

elif strona == "13. PageIndex":
    from pages import pageindex
    pageindex.render()

elif strona == "14. Agentic RAG":
    from pages import agentic_rag
    agentic_rag.render()

elif strona == "15. Graph RAG":
    from pages import graph_rag
    graph_rag.render()

elif strona == "16. Hybrid Vector+Graph":
    from pages import hybrid_vg
    hybrid_vg.render()

elif strona == "17. Context Engineering":
    from pages import context_eng
    context_eng.render()

elif strona == "18. RAG-Fusion":
    from pages import rag_fusion
    rag_fusion.render()

elif strona == "19. FLARE":
    from pages import flare
    flare.render()

elif strona == "20. RAPTOR":
    from pages import raptor
    raptor.render()

elif strona == "21. Speculative RAG":
    from pages import speculative
    speculative.render()

elif strona == "22. Decision Framework":
    from pages import decision_framework
    decision_framework.render()
