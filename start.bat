@echo off
chcp 65001 >nul
title RAG Pipeline - Neural Lab
cd /d "%~dp0"

echo.
echo ============================================================
echo   RAG Pipeline - Neural Lab (22 techniki RAG)
echo ============================================================
echo.

REM ==== Sprawdz Python ====
python --version >nul 2>&1
if errorlevel 1 (
    echo [BLAD] Python nie jest zainstalowany lub brak w PATH.
    echo.
    echo Pobierz Python 3.12+ z: https://www.python.org/downloads/
    echo Podczas instalacji zaznacz "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% znaleziony.

REM ==== Srodowisko wirtualne ====
if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo [SETUP] Pierwsze uruchomienie - tworze srodowisko .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [BLAD] Nie mozna utworzyc venv. Sprawdz instalacje Python.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo [SETUP] Instalacja zaleznosci (to moze potrwac 3-5 minut)...
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [BLAD] Blad instalacji zaleznosci.
        pause
        exit /b 1
    )
    echo [OK] Srodowisko gotowe.
) else (
    call .venv\Scripts\activate.bat
    echo [OK] Srodowisko .venv aktywne.
)

REM ==== Plik .env ====
if not exist ".env" (
    if exist ".env.example" (
        echo.
        echo ============================================================
        echo   KONFIGURACJA API - PIERWSZE URUCHOMIENIE
        echo ============================================================
        copy .env.example .env >nul
        echo [SETUP] Utworzono plik .env (na bazie .env.example).
        echo.
        echo Musisz uzupelnic klucze API przed uzyciem pipeline:
        echo   - QDRANT_URL, QDRANT_API_KEY (https://cloud.qdrant.io/)
        echo   - JINA_API_KEY              (https://jina.ai/)
        echo   - ANTHROPIC_API_KEY         (https://console.anthropic.com/)
        echo.
        echo Opcjonalne:
        echo   - GOOGLE_API_KEY            (benchmarki Gemini)
        echo   - NEO4J_URI + PASSWORD      (Graph RAG)
        echo.
        echo Edytuj plik .env notatnikiem i uzupelnij klucze.
        echo Po uzupelnieniu zamknij edytor i nacisnij dowolny klawisz.
        echo.
        notepad .env
        pause
    )
)

REM ==== Uruchom Streamlit + otworz przegladarke ====
echo.
echo ============================================================
echo   RAG Pipeline uruchomiony
echo   URL: http://localhost:8501
echo   Zamknij to okno aby zatrzymac serwer.
echo ============================================================
echo.

REM Auto-open browser po 3 sekundach (w tle)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

pause
