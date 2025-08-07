#!/usr/bin/env python3
"""
Script de démonstration du système de reconnaissance faciale
Utilise la webcam pour une démonstration en temps réel
"""

import cv2
import asyncio
import logging
import time
from pathlib import Path
import sys
import os

# Ajouter le chemin vers le module principal
sys.path.insert(0, str(Path(__file__).parent))

from app.core.face_engine import FaceRecognitionEngine
from app.services.database import create_database_service
from app.core.config import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Exécute une commande et affiche le résultat"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"Commande: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        
        if result.stdout:
            print("📤 Sortie:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("❌ Erreur:")
            print(result.stderr)
            return False
        
        if result.returncode == 0:
            print("✅ Succès!")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return False

def check_requirements():
    """Vérifie que les dépendances sont installées"""
    print("🔍 Vérification des dépendances...")
    
    required_packages = ["cv2", "numpy", "yaml", "faiss", "insightface"]
    missing = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "numpy":
                import numpy
            elif package == "yaml":
                import yaml
            elif package == "faiss":
                import faiss
            elif package == "insightface":
                import insightface
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n⚠️  Packages manquants: {missing}")
        print("Installez avec: pip install -r requirements.txt")
        return False
    
    print("\n✅ Toutes les dépendances sont installées!")
    return True

def check_data_structure():
    """Vérifie la structure des données"""
    print("\n🗂️  Vérification de la structure des données...")
    
    people_dir = Path("data/people")
    if not people_dir.exists():
        print("❌ Dossier data/people manquant")
        return False
    
    people_found = []
    for person_dir in people_dir.iterdir():
        if person_dir.is_dir():
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
            if images:
                people_found.append((person_dir.name, len(images)))
                print(f"✅ {person_dir.name}: {len(images)} images")
    
    if not people_found:
        print("⚠️  Aucune personne avec des images trouvée dans data/people/")
        print("Ajoutez des photos dans data/people/<nom>/")
        return False
    
    print(f"\n✅ {len(people_found)} personnes trouvées avec des images")
    return True

def main():
    parser = argparse.ArgumentParser(description="Démonstration du système Face-ID")
    parser.add_argument("--skip-checks", action="store_true", help="Ignorer les vérifications")
    parser.add_argument("--index-only", action="store_true", help="Créer seulement l'index")
    parser.add_argument("--test-image", help="Tester sur une image spécifique")
    parser.add_argument("--calibrate", action="store_true", help="Calibrer le seuil")
    args = parser.parse_args()
    
    print("🎯 Démonstration du système Face-ID")
    print("=" * 50)
    
    # Vérifications préliminaires
    if not args.skip_checks:
        if not check_requirements():
            return
        
        if not check_data_structure():
            print("\n💡 Pour commencer:")
            print("1. Créez des dossiers dans data/people/ (ex: data/people/alice/)")
            print("2. Ajoutez 3-10 photos par personne")
            print("3. Relancez cette démonstration")
            return
    
    # Étape 1: Création de l'index
    success = run_command(
        "python scripts/index_people.py",
        "Création de l'index FAISS à partir des photos"
    )
    
    if not success:
        print("❌ Échec de la création de l'index")
        return
    
    if args.index_only:
        print("\n🎉 Index créé avec succès!")
        return
    
    # Étape 2: Calibration (optionnel)
    if args.calibrate:
        print("\n📊 Calibration du seuil de reconnaissance...")
        run_command(
            "python scripts/calibrate_threshold.py",
            "Calibration automatique du seuil"
        )
    
    # Étape 3: Test sur image (si fournie)
    if args.test_image:
        if Path(args.test_image).exists():
            run_command(
                f"python scripts/identify_image.py --img \"{args.test_image}\"",
                f"Test de reconnaissance sur {args.test_image}"
            )
        else:
            print(f"❌ Image {args.test_image} non trouvée")
    
    # Étape 4: Test vidéo
    print("\n🎥 Démarrage du test vidéo...")
    print("Appuyez sur ESC pour quitter")
    
    run_command(
        "python scripts/identify_video.py --video 0",
        "Test de reconnaissance en temps réel (webcam)"
    )
    
    print("\n🎉 Démonstration terminée!")
    print("\n📖 Commandes utiles:")
    print("- Test image: python scripts/identify_image.py --img photo.jpg")
    print("- Test vidéo: python scripts/identify_video.py --video 0")
    print("- Calibration: python scripts/calibrate_threshold.py")
    print("- Tests unitaires: python -m pytest tests/")

if __name__ == "__main__":
    # Changement vers le répertoire du script pour les chemins relatifs
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    main()
