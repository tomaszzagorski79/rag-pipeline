"""RAG Pipeline — główna aplikacja Streamlit (Neural Lab theme).

Uruchomienie:
    .venv\\Scripts\\streamlit run app.py
"""

from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="RAG Pipeline — Neural Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (Neural Lab theme) ---
_css_path = Path(__file__).parent / "assets" / "custom.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# --- Nawigacja z ikonami ---
NAV_ITEMS = [
    ("🏠", "Przegląd"),
    ("📥", "1. Scraping"),
    ("🔤", "2. Embeddingi tytułów"),
    ("📑", "3. Chunking & Indeksowanie"),
    ("🔍", "4. Zapytania"),
    ("🎯", "5. Re-ranking"),
    ("💭", "6. HyDE"),
    ("🧠", "7. Adaptive RAG"),
    ("🔄", "8. CRAG"),
    ("📊", "9. Ewaluacja RAGAS"),
    ("🔬", "10. Hallucination Detection"),
    ("⚡", "11. Benchmarki embeddingów"),
    ("💎", "12. CRQ Scoring"),
    ("📖", "13. PageIndex"),
    ("🤖", "14. Agentic RAG"),
    ("🕸️", "15. Graph RAG"),
    ("⚛️", "16. Hybrid Vector+Graph"),
    ("🎛️", "17. Context Engineering"),
    ("🔀", "18. RAG-Fusion"),
    ("🔥", "19. FLARE"),
    ("🌲", "20. RAPTOR"),
    ("🏇", "21. Speculative RAG"),
    ("🗺️", "22. Decision Framework"),
]

NAV_LABELS = [f"{icon}  {name}" for icon, name in NAV_ITEMS]
NAV_LABEL_TO_NAME = {label: name for label, (_, name) in zip(NAV_LABELS, NAV_ITEMS)}

st.sidebar.markdown(
    """
    <div style="padding: 12px 0 4px 0;">
        <h1 style="margin: 0; font-size: 1.4em; font-weight: 800;">
            🧠 RAG Pipeline
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 0.75em; color: #71717A; letter-spacing: 0.05em; text-transform: uppercase;">
            Neural Lab · 22 Techniki
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

strona_label = st.sidebar.radio(
    "Nawigacja",
    NAV_LABELS,
    label_visibility="collapsed",
)
strona = NAV_LABEL_TO_NAME[strona_label]

# --- Breadcrumb ---
current_icon = next((icon for icon, name in NAV_ITEMS if name == strona), "🔍")
st.markdown(
    f"""
    <div style="
        display: flex; align-items: center; gap: 8px;
        padding: 8px 0 16px 0;
        font-size: 0.85em; color: #71717A;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 20px;
    ">
        <span style="opacity: 0.6;">RAG Pipeline</span>
        <span style="opacity: 0.4;">/</span>
        <span style="color: #A78BFA; font-weight: 500;">{current_icon} {strona}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status .env
st.sidebar.markdown("---")
st.sidebar.caption("Status konfiguracji")
try:
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent / ".env", override=True)
    keys = {
        "QDRANT_URL": (bool(os.getenv("QDRANT_URL")), True),
        "QDRANT_API_KEY": (bool(os.getenv("QDRANT_API_KEY")), True),
        "JINA_API_KEY": (bool(os.getenv("JINA_API_KEY")), True),
        "ANTHROPIC_API_KEY": (bool(os.getenv("ANTHROPIC_API_KEY")), True),
        "GOOGLE_API_KEY": (bool(os.getenv("GOOGLE_API_KEY")), False),
        "NEO4J_URI": (bool(os.getenv("NEO4J_URI")), False),
    }
    for name, (ok, required) in keys.items():
        icon = "✅" if ok else ("❌" if required else "⬜")
        suffix = "" if required else " (opcj.)"
        st.sidebar.text(f"{icon} {name}{suffix}")

    # Link do setup wizarda jeśli brakuje wymaganych
    missing_req = [n for n, (ok, req) in keys.items() if req and not ok]
    if missing_req:
        st.sidebar.error(f"⚠️ Brakuje {len(missing_req)} wymaganych kluczy — zobacz zakładkę **Przegląd**")
except Exception:
    st.sidebar.error("Brak pliku .env — skopiuj .env.example → .env")


# --- Ostrzeżenie gdy strona wymaga klucza którego brak ---
def _require_qdrant():
    """Blokada stron wymagających Qdrant."""
    import os
    if not (os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY")):
        st.error(
            "🔒 **Ta zakładka wymaga Qdrant Cloud.** "
            "Dodaj `QDRANT_URL` i `QDRANT_API_KEY` do `.env` "
            "(darmowy tier: [cloud.qdrant.io](https://cloud.qdrant.io/)). "
            "Szczegóły w zakładce **Przegląd** → Setup API."
        )
        st.stop()


def _require_neo4j():
    """Blokada stron wymagających Neo4j."""
    import os
    if not (os.getenv("NEO4J_URI") and os.getenv("NEO4J_PASSWORD")):
        st.error(
            "🔒 **Ta zakładka wymaga Neo4j Aura.** "
            "Dodaj `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` do `.env` "
            "(darmowy tier: [console.neo4j.io](https://console.neo4j.io/))."
        )
        st.stop()


# Mapa zakładek wymagających konkretnych kluczy
STRONY_WYMAGAJACE_QDRANT = {
    "2. Embeddingi tytułów", "3. Chunking & Indeksowanie", "4. Zapytania",
    "5. Re-ranking", "6. HyDE", "7. Adaptive RAG", "8. CRAG",
    "9. Ewaluacja RAGAS", "14. Agentic RAG", "16. Hybrid Vector+Graph",
    "18. RAG-Fusion", "19. FLARE", "21. Speculative RAG",
}
STRONY_WYMAGAJACE_NEO4J = {
    "15. Graph RAG", "16. Hybrid Vector+Graph",
}

if strona in STRONY_WYMAGAJACE_QDRANT:
    _require_qdrant()
if strona in STRONY_WYMAGAJACE_NEO4J:
    _require_neo4j()


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
