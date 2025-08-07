"""
Application FastAPI simplifiée pour tests de démarrage
Utilise OpenCV au lieu d'InsightFace
"""

import os
import io
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import asyncio

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importation des modules locaux
try:
    from app.core.config import Config
    from app.core.face_engine_simple import FaceRecognitionEngine
    from app.services.database import create_database_service
    from app.models.schemas import (
        FaceSearchRequest, FaceSearchResponse, 
        PersonEnrollRequest, PersonEnrollResponse,
        FaceVerificationRequest, FaceVerificationResponse
    )
except ImportError as e:
    logger.error(f"Erreur d'importation : {e}")
    # Créer des classes temporaires pour le développement
    class Config:
        def __init__(self):
            self.api = type('api', (), {
                'host': '0.0.0.0',
                'port': 8000,
                'max_upload_size': 50,
                'cors_origins': ['*']
            })()
            self.thresholds = type('thresholds', (), {
                'cosine_similarity': 0.35,
                'liveness_score': 0.7,
                'face_quality': 0.5
            })()

# Configuration globale
config = Config()

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Face Recognition API",
    description="API de reconnaissance faciale - Version de développement",
    version="0.1.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour les services
face_engine = None
database_service = None

# Templates et fichiers statiques
templates = Jinja2Templates(directory="web/templates")
app.mount("/web/static", StaticFiles(directory="web/static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'application"""
    global face_engine, database_service
    
    try:
        logger.info("🚀 Initialisation des services...")
        
        # Créer les dossiers nécessaires
        folders = ["storage/uploads", "storage/database", "storage/temp", "storage/logs"]
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        # Initialiser le moteur de reconnaissance faciale
        face_engine = FaceRecognitionEngine(config)
        logger.info("✅ Moteur de reconnaissance faciale initialisé")
        
        # Initialiser la base de données (version simple en mémoire pour les tests)
        database_service = SimpleDatabase()
        logger.info("✅ Base de données initialisée")
        
        logger.info("🎉 Tous les services sont prêts !")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation : {e}")
        raise

class SimpleDatabase:
    """Base de données simple en mémoire pour les tests"""
    
    def __init__(self):
        self.persons = {}
        self.embeddings = {}
        self.next_id = 1
    
    async def add_person(self, name: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Ajoute une nouvelle personne"""
        person_id = str(self.next_id)
        self.next_id += 1
        
        self.persons[person_id] = {
            'id': person_id,
            'name': name,
            'metadata': metadata,
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        self.embeddings[person_id] = embedding
        return person_id
    
    async def search_similar(self, embedding: np.ndarray, top_k: int = 10, threshold: float = 0.35) -> List[Dict]:
        """Recherche les visages similaires"""
        results = []
        
        for person_id, stored_embedding in self.embeddings.items():
            # Calculer la similarité cosinus
            norm1 = np.linalg.norm(embedding)
            norm2 = np.linalg.norm(stored_embedding)
            
            if norm1 == 0 or norm2 == 0:
                continue
                
            similarity = np.dot(embedding, stored_embedding) / (norm1 * norm2)
            
            if similarity > threshold:
                person = self.persons[person_id]
                results.append({
                    'person_id': person_id,
                    'person_name': person['name'],
                    'similarity': float(similarity),
                    'metadata': person['metadata']
                })
        
        # Trier par similarité décroissante
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    async def get_person(self, person_id: str) -> Optional[Dict]:
        """Récupère les informations d'une personne"""
        return self.persons.get(person_id)
    
    async def list_persons(self) -> List[Dict]:
        """Liste toutes les personnes"""
        return list(self.persons.values())
    
    async def delete_person(self, person_id: str) -> bool:
        """Supprime une personne"""
        if person_id in self.persons:
            del self.persons[person_id]
            del self.embeddings[person_id]
            return True
        return False

def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Charge une image depuis un fichier uploadé"""
    try:
        # Lire le contenu du fichier
        content = file.file.read()
        
        # Convertir en image PIL
        pil_image = Image.open(io.BytesIO(content))
        
        # Convertir en RGB si nécessaire
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convertir en array numpy pour OpenCV
        image_array = np.array(pil_image)
        
        # OpenCV utilise BGR, PIL utilise RGB
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'image : {e}")
        raise HTTPException(status_code=400, detail=f"Impossible de charger l'image : {e}")

# Routes de l'API

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil qui redirige vers l'interface web"""
    return """
    <html>
        <head>
            <title>Face Recognition API</title>
        </head>
        <body>
            <h1>🎭 Face Recognition API</h1>
            <p>Bienvenue dans l'API de reconnaissance faciale !</p>
            <ul>
                <li><a href="/web">🌐 Interface Web</a></li>
                <li><a href="/docs">📖 Documentation API</a></li>
                <li><a href="/health">💚 État de santé</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/web", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Interface web principale"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"<h1>Erreur</h1><p>Impossible de charger l'interface web : {e}</p>")

@app.get("/health")
async def health_check():
    """Vérification de l'état de santé de l'API"""
    return {
        "status": "healthy",
        "service": "Face Recognition API",
        "version": "0.1.0",
        "face_engine": "OpenCV" if face_engine and face_engine.is_initialized else "Non initialisé",
        "database": "Memory" if database_service else "Non initialisé"
    }

@app.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    source: str = Form("upload"),
    consent_type: str = Form("explicit"),
    files: List[UploadFile] = File(...)
):
    """Enregistre une nouvelle personne avec ses photos"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="Aucun fichier fourni")
        
        logger.info(f"📝 Enregistrement de {name} avec {len(files)} photo(s)")
        
        # Traiter toutes les images et extraire les embeddings
        embeddings = []
        for i, file in enumerate(files):
            # Charger l'image
            image = load_image_from_upload(file)
            
            # Détecter les visages
            faces = face_engine.detect_faces(image)
            
            if not faces:
                logger.warning(f"⚠️ Aucun visage détecté dans l'image {i+1}")
                continue
            
            # Prendre le premier visage détecté
            face = faces[0]
            embeddings.append(face['embedding'])
        
        if not embeddings:
            raise HTTPException(status_code=400, detail="Aucun visage détecté dans les images fournies")
        
        # Calculer l'embedding moyen
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Enregistrer dans la base de données
        person_id = await database_service.add_person(
            name=name,
            embedding=avg_embedding,
            metadata={
                'source': source,
                'consent_type': consent_type,
                'num_images': len(embeddings)
            }
        )
        
        logger.info(f"✅ Personne {name} enregistrée avec l'ID {person_id}")
        
        return {
            "success": True,
            "person_id": person_id,
            "name": name,
            "images_processed": len(embeddings),
            "message": f"Personne {name} enregistrée avec succès"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

@app.post("/search")
async def search_faces(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    threshold: float = Form(0.35)
):
    """Recherche des visages similaires dans la base de données"""
    try:
        logger.info(f"🔍 Recherche avec seuil {threshold} et top_k {top_k}")
        
        # Charger l'image
        image = load_image_from_upload(file)
        
        # Détecter les visages
        faces = face_engine.detect_faces(image)
        
        if not faces:
            return {
                "success": False,
                "message": "Aucun visage détecté dans l'image",
                "results": []
            }
        
        # Prendre le premier visage détecté
        face = faces[0]
        embedding = face['embedding']
        
        # Rechercher dans la base de données
        results = await database_service.search_similar(
            embedding=embedding,
            top_k=top_k,
            threshold=threshold
        )
        
        logger.info(f"✅ Trouvé {len(results)} correspondance(s)")
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "results": results,
            "message": f"Recherche terminée - {len(results)} résultat(s) trouvé(s)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

@app.post("/verify")
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """Vérifie si deux images contiennent la même personne"""
    try:
        logger.info("🔄 Vérification de deux visages")
        
        # Charger les deux images
        image1 = load_image_from_upload(file1)
        image2 = load_image_from_upload(file2)
        
        # Détecter les visages dans chaque image
        faces1 = face_engine.detect_faces(image1)
        faces2 = face_engine.detect_faces(image2)
        
        if not faces1:
            raise HTTPException(status_code=400, detail="Aucun visage détecté dans la première image")
        
        if not faces2:
            raise HTTPException(status_code=400, detail="Aucun visage détecté dans la deuxième image")
        
        # Prendre le premier visage de chaque image
        embedding1 = faces1[0]['embedding']
        embedding2 = faces2[0]['embedding']
        
        # Vérifier la correspondance
        is_same, similarity = face_engine.verify_faces(embedding1, embedding2)
        
        logger.info(f"✅ Vérification terminée - Similarité: {similarity:.3f}")
        
        return {
            "success": True,
            "is_same_person": is_same,
            "similarity": similarity,
            "threshold": config.thresholds.cosine_similarity,
            "message": "Même personne" if is_same else "Personnes différentes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

@app.get("/persons")
async def list_persons():
    """Liste toutes les personnes enregistrées"""
    try:
        persons = await database_service.list_persons()
        return {
            "success": True,
            "count": len(persons),
            "persons": persons
        }
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des personnes : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

@app.get("/person/{person_id}")
async def get_person(person_id: str):
    """Récupère les informations d'une personne"""
    try:
        person = await database_service.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Personne non trouvée")
        
        return {
            "success": True,
            "person": person
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération de la personne : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

@app.delete("/person/{person_id}")
async def delete_person(person_id: str):
    """Supprime une personne (conformité RGPD)"""
    try:
        success = await database_service.delete_person(person_id)
        if not success:
            raise HTTPException(status_code=404, detail="Personne non trouvée")
        
        logger.info(f"🗑️ Personne {person_id} supprimée")
        
        return {
            "success": True,
            "message": f"Personne {person_id} supprimée avec succès"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la suppression : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

@app.get("/stats")
async def get_stats():
    """Statistiques du système"""
    try:
        persons = await database_service.list_persons()
        
        return {
            "success": True,
            "statistics": {
                "total_persons": len(persons),
                "face_engine": "OpenCV Haar Cascade",
                "database_type": "Memory",
                "version": "0.1.0-dev"
            }
        }
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

# Export de l'app pour uvicorn
app_instance = app
