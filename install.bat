@echo off
echo === Installation du systeme Face-ID ===

REM Verification de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

echo Python detecte
python --version

REM Creation de l'environnement virtuel
echo Creation de l'environnement virtuel...
python -m venv .venv

REM Activation de l'environnement virtuel
echo Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat

REM Installation des dependances
echo Installation des dependances...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo === Installation terminee ===
echo.
echo Pour utiliser le systeme:
echo 1. Activez l'environnement virtuel: .venv\Scripts\activate.bat
echo 2. Placez des photos dans data/people/^<nom^>/
echo 3. Creez l'index: python scripts/index_people.py
echo 4. Testez: python scripts/identify_video.py --video 0
echo.
pause
