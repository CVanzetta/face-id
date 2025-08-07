#!/usr/bin/env python3
"""
Point d'entrée principal pour l'API de reconnaissance faciale
Version temporaire utilisant OpenCV uniquement
"""

import sys
import logging
from pathlib import Path
import uvicorn

# Ajouter le répertoire du projet au Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Lance l'application FastAPI"""
    try:
        # Créer les dossiers nécessaires
        from app.core.config import Config
        
        config = Config()
        
        # Créer la structure de dossiers
        folders = [
            "storage/uploads",
            "storage/database", 
            "storage/temp",
            "storage/logs",
            "storage/models"
        ]
        
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 Démarrage de l'API de reconnaissance faciale (mode développement)")
        logger.info(f"📂 Répertoire de travail : {project_root}")
        logger.info(f"🌐 Interface web : http://localhost:{config.api.port}/web")
        logger.info(f"📖 Documentation API : http://localhost:{config.api.port}/docs")
        
        # Configuration uvicorn
        uvicorn.run(
            "app_simple:app",
            host=config.api.host,
            port=config.api.port,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Arrêt de l'API demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
