# Script d'installation pour Windows
# Exécutez ce script avec PowerShell

Write-Host "=== Installation du système Face-ID ===" -ForegroundColor Green

# Vérification de Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python détecté: $pythonVersion" -ForegroundColor Blue
} catch {
    Write-Host "ERREUR: Python n'est pas installé ou pas dans le PATH" -ForegroundColor Red
    exit 1
}

# Création de l'environnement virtuel
Write-Host "Création de l'environnement virtuel..." -ForegroundColor Yellow
python -m venv .venv

# Activation de l'environnement virtuel
Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Installation des dépendances
Write-Host "Installation des dépendances..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "=== Installation terminée ===" -ForegroundColor Green
Write-Host ""
Write-Host "Pour utiliser le système:" -ForegroundColor Cyan
Write-Host "1. Activez l'environnement virtuel: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Placez des photos dans data/people/<nom>/" -ForegroundColor White
Write-Host "3. Créez l'index: python scripts/index_people.py" -ForegroundColor White
Write-Host "4. Testez: python scripts/identify_video.py --video 0" -ForegroundColor White
