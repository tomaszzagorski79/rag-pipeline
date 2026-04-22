@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   RAG Pipeline - Neural Lab (22 techniki RAG)
echo ============================================================
echo.

REM ==== Sprawdz Python ====
where python >nul 2>nul
if errorlevel 1 goto :no_python

python --version
echo [OK] Python znaleziony.
echo.

REM ==== Srodowisko wirtualne ====
if exist ".venv\Scripts\activate.bat" goto :venv_exists

echo [SETUP] Pierwsze uruchomienie - tworze srodowisko .venv...
python -m venv .venv
if errorlevel 1 goto :venv_error

call .venv\Scripts\activate.bat
echo [SETUP] Instalacja zaleznosci ^(3-5 minut^)...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 goto :pip_error

echo [OK] Srodowisko gotowe.
goto :check_env

:venv_exists
call .venv\Scripts\activate.bat
echo [OK] Srodowisko .venv aktywne.

:check_env
REM ==== Plik .env ====
if exist ".env" goto :run_app
if not exist ".env.example" goto :run_app

echo.
echo ============================================================
echo   KONFIGURACJA API - PIERWSZE URUCHOMIENIE
echo ============================================================
copy .env.example .env >nul
echo [SETUP] Utworzono plik .env.
echo Uzupelnij klucze API w notatniku, zapisz i zamknij.
echo.
notepad .env

:run_app
echo.
echo ============================================================
echo   RAG Pipeline uruchomiony
echo   URL: http://localhost:8501
echo   Zamknij to okno aby zatrzymac serwer.
echo ============================================================
echo.

start "" "http://localhost:8501"
python -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
goto :end

:no_python
echo [BLAD] Python nie jest zainstalowany lub brak w PATH.
echo Pobierz Python 3.12+ z: https://www.python.org/downloads/
echo Podczas instalacji zaznacz "Add Python to PATH".
pause
exit /b 1

:venv_error
echo [BLAD] Nie mozna utworzyc venv.
pause
exit /b 1

:pip_error
echo [BLAD] Blad instalacji zaleznosci.
pause
exit /b 1

:end
pause
