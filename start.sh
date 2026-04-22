#!/bin/bash
# RAG Pipeline - Neural Lab (22 techniki RAG)
# Uruchomienie: ./start.sh lub dwuklik w Finder (jako start.command)

cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  RAG Pipeline - Neural Lab (22 techniki RAG)"
echo "============================================================"
echo ""

# ==== Sprawdz Python ====
if ! command -v python3 &> /dev/null; then
    echo "[BLAD] Python 3 nie jest zainstalowany."
    echo ""
    echo "Instalacja:"
    echo "  Mac:    brew install python@3.12"
    echo "  Linux:  sudo apt install python3.12 python3.12-venv"
    echo ""
    read -p "Nacisnij Enter aby zamknac..."
    exit 1
fi
PY_VERSION=$(python3 --version | awk '{print $2}')
echo "[OK] Python ${PY_VERSION} znaleziony."

# ==== Srodowisko wirtualne ====
if [ ! -d ".venv" ]; then
    echo ""
    echo "[SETUP] Pierwsze uruchomienie - tworze srodowisko .venv..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[BLAD] Nie mozna utworzyc venv."
        read -p "Nacisnij Enter..."
        exit 1
    fi
    source .venv/bin/activate
    echo "[SETUP] Instalacja zaleznosci (to moze potrwac 3-5 minut)..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[BLAD] Blad instalacji zaleznosci."
        read -p "Nacisnij Enter..."
        exit 1
    fi
    echo "[OK] Srodowisko gotowe."
else
    source .venv/bin/activate
    echo "[OK] Srodowisko .venv aktywne."
fi

# ==== Plik .env ====
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo ""
        echo "============================================================"
        echo "  KONFIGURACJA API - PIERWSZE URUCHOMIENIE"
        echo "============================================================"
        cp .env.example .env
        echo "[SETUP] Utworzono plik .env (na bazie .env.example)."
        echo ""
        echo "Musisz uzupelnic klucze API przed uzyciem pipeline:"
        echo "  - QDRANT_URL, QDRANT_API_KEY (https://cloud.qdrant.io/)"
        echo "  - JINA_API_KEY              (https://jina.ai/)"
        echo "  - ANTHROPIC_API_KEY         (https://console.anthropic.com/)"
        echo ""
        echo "Opcjonalne:"
        echo "  - GOOGLE_API_KEY            (benchmarki Gemini)"
        echo "  - NEO4J_URI + PASSWORD      (Graph RAG)"
        echo ""

        # Otworz edytor (nano na Linux, TextEdit na Mac)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open -e .env
        elif command -v nano &> /dev/null; then
            nano .env
        elif command -v vim &> /dev/null; then
            vim .env
        else
            echo "Otworz plik .env recznie i uzupelnij klucze."
        fi

        read -p "Po uzupelnieniu .env nacisnij Enter..."
    fi
fi

# ==== Uruchom Streamlit ====
echo ""
echo "============================================================"
echo "  RAG Pipeline uruchomiony"
echo "  URL: http://localhost:8501"
echo "  Ctrl+C aby zatrzymac serwer."
echo "============================================================"
echo ""

# Auto-open browser po 3 sekundach (w tle)
(sleep 3 && {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:8501
    elif command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8501
    fi
}) &

python -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
