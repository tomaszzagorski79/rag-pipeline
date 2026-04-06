@echo off
chcp 65001 >nul
title RAG Pipeline

cd "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    echo Aktywuje istniejace srodowisko .venv...
    call .venv\Scripts\activate.bat
) else (
    echo Tworze srodowisko .venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
)

echo.
echo  RAG Pipeline
echo  http://localhost:8501
echo  Zamknij to okno aby zatrzymac serwer.
echo.

streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
pause
