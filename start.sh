#!/bin/bash
# RAG Pipeline - uruchomienie na Mac / Linux

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Tworze srodowisko .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    echo "Aktywuje istniejace srodowisko .venv..."
    source .venv/bin/activate
fi

if [ ! -f ".env" ]; then
    echo "UWAGA: Brak pliku .env. Skopiuj .env.example do .env i uzupelnij klucze API."
    cp .env.example .env
    exit 1
fi

echo ""
echo "  RAG Pipeline"
echo "  http://localhost:8501"
echo "  Ctrl+C aby zatrzymac serwer."
echo ""

streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
