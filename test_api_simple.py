#!/usr/bin/env python3
"""
API de reconnaissance faciale ultra-simplifiée pour test de démarrage
Tout-en-un pour éviter les problèmes d'import
"""

import os
import io
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration globale
class SimpleConfig:
    def __init__(self):
        self.api_host = '0.0.0.0'
        self.api_port = 8000
        self.cosine_threshold = 0.35

config = SimpleConfig()

# Base de données simple en mémoire
class MemoryDatabase:
    def __init__(self):
        self.persons = {}
        self.embeddings = {}
        self.next_id = 1
    
    def add_person(self, name: str, embedding: np.ndarray, metadata: Dict = None) -> str:
        person_id = str(self.next_id)
        self.next_id += 1
        
        self.persons[person_id] = {
            'id': person_id,
            'name': name,
            'metadata': metadata or {},
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        self.embeddings[person_id] = embedding
        logger.info(f"✅ Ajouté {name} avec l'ID {person_id}")
        return person_id
    
    def search_similar(self, embedding: np.ndarray, top_k: int = 10, threshold: float = 0.35) -> List[Dict]:
        results = []
        
        for person_id, stored_embedding in self.embeddings.items():
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
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_person(self, person_id: str) -> Optional[Dict]:
        return self.persons.get(person_id)
    
    def list_persons(self) -> List[Dict]:
        return list(self.persons.values())
    
    def delete_person(self, person_id: str) -> bool:
        if person_id in self.persons:
            del self.persons[person_id]
            del self.embeddings[person_id]
            return True
        return False

# Moteur de reconnaissance simple
class SimpleFaceEngine:
    def __init__(self):
        self.face_cascade = None
        self.is_initialized = False
        self._init_detector()
    
    def _init_detector(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Impossible de charger le détecteur")
            
            logger.info("✅ Détecteur OpenCV initialisé")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Erreur détecteur : {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if not self.is_initialized:
            return []
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            detected_faces = []
            for (x, y, w, h) in faces:
                face_region = image[y:y+h, x:x+w]
                embedding = self._generate_embedding(face_region)
                
                face_data = {
                    'bbox': [x, y, x+w, y+h],
                    'embedding': embedding,
                    'confidence': 0.8
                }
                detected_faces.append(face_data)
            
            logger.debug(f"🔍 Détecté {len(detected_faces)} visage(s)")
            return detected_faces
            
        except Exception as e:
            logger.error(f"❌ Erreur détection : {e}")
            return []
    
    def _generate_embedding(self, face_region: np.ndarray) -> np.ndarray:
        try:
            # Simple embedding basé sur la taille et les pixels moyens
            face_resized = cv2.resize(face_region, (32, 32))
            
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Normaliser
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Caractéristiques simples
            hist = cv2.calcHist([face_gray], [0], None, [8], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-7)
            
            # Moyennes par quadrants
            h, w = face_gray.shape
            quad1 = np.mean(face_gray[0:h//2, 0:w//2])
            quad2 = np.mean(face_gray[0:h//2, w//2:w])
            quad3 = np.mean(face_gray[h//2:h, 0:w//2])
            quad4 = np.mean(face_gray[h//2:h, w//2:w])
            
            # Combiner les caractéristiques
            embedding = np.concatenate([
                hist,
                [quad1, quad2, quad3, quad4]
            ])
            
            # Normaliser
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Erreur embedding : {e}")
            return np.random.rand(12).astype(np.float32)
    
    def verify_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> tuple:
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return False, 0.0
                
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            is_same = similarity > config.cosine_threshold
            
            return is_same, float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification : {e}")
            return False, 0.0

# Initialisation des services globaux
face_engine = SimpleFaceEngine()
database = MemoryDatabase()

# Initialisation FastAPI
app = FastAPI(
    title="Face Recognition API - Test",
    description="API de test pour la reconnaissance faciale",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_image_from_upload(file: UploadFile) -> np.ndarray:
    try:
        content = file.file.read()
        pil_image = Image.open(io.BytesIO(content))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        image_array = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        logger.error(f"Erreur chargement image : {e}")
        raise HTTPException(status_code=400, detail=f"Image invalide : {e}")

# Routes de l'API

@app.get("/")
async def root():
    return {
        "message": "🎭 Face Recognition API - Version Test",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "enroll": "/enroll",
            "search": "/search", 
            "verify": "/verify",
            "persons": "/persons",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Face Recognition API",
        "version": "0.1.0-test",
        "face_engine": "OpenCV" if face_engine.is_initialized else "Error",
        "database": "Memory",
        "total_persons": len(database.persons)
    }

@app.post("/enroll")
async def enroll_person(
    name: str = Form(...),
    source: str = Form("upload"),
    consent_type: str = Form("explicit"),
    files: List[UploadFile] = File(...)
):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="Aucun fichier fourni")
        
        logger.info(f"📝 Enregistrement de {name} avec {len(files)} photo(s)")
        
        embeddings = []
        for i, file in enumerate(files):
            image = load_image_from_upload(file)
            faces = face_engine.detect_faces(image)
            
            if not faces:
                logger.warning(f"⚠️ Aucun visage dans l'image {i+1}")
                continue
            
            face = faces[0]
            embeddings.append(face['embedding'])
        
        if not embeddings:
            raise HTTPException(status_code=400, detail="Aucun visage détecté")
        
        # Moyenne des embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        person_id = database.add_person(
            name=name,
            embedding=avg_embedding,
            metadata={
                'source': source,
                'consent_type': consent_type,
                'num_images': len(embeddings)
            }
        )
        
        return {
            "success": True,
            "person_id": person_id,
            "name": name,
            "images_processed": len(embeddings),
            "message": f"Personne {name} enregistrée avec l'ID {person_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur enregistrement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

@app.post("/search")
async def search_faces(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    threshold: float = Form(0.35)
):
    try:
        logger.info(f"🔍 Recherche avec seuil {threshold}")
        
        image = load_image_from_upload(file)
        faces = face_engine.detect_faces(image)
        
        if not faces:
            return {
                "success": False,
                "message": "Aucun visage détecté",
                "results": []
            }
        
        face = faces[0]
        embedding = face['embedding']
        
        results = database.search_similar(
            embedding=embedding,
            top_k=top_k,
            threshold=threshold
        )
        
        logger.info(f"✅ Trouvé {len(results)} correspondance(s)")
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "results": results,
            "message": f"{len(results)} résultat(s) trouvé(s)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur recherche : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

@app.post("/verify")
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    try:
        logger.info("🔄 Vérification de deux visages")
        
        image1 = load_image_from_upload(file1)
        image2 = load_image_from_upload(file2)
        
        faces1 = face_engine.detect_faces(image1)
        faces2 = face_engine.detect_faces(image2)
        
        if not faces1:
            raise HTTPException(status_code=400, detail="Aucun visage dans l'image 1")
        
        if not faces2:
            raise HTTPException(status_code=400, detail="Aucun visage dans l'image 2")
        
        embedding1 = faces1[0]['embedding']
        embedding2 = faces2[0]['embedding']
        
        is_same, similarity = face_engine.verify_faces(embedding1, embedding2)
        
        logger.info(f"✅ Similarité: {similarity:.3f}")
        
        return {
            "success": True,
            "is_same_person": is_same,
            "similarity": similarity,
            "threshold": config.cosine_threshold,
            "message": "Même personne" if is_same else "Personnes différentes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur vérification : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

@app.get("/persons")
async def list_persons():
    try:
        persons = database.list_persons()
        return {
            "success": True,
            "count": len(persons),
            "persons": persons
        }
    except Exception as e:
        logger.error(f"❌ Erreur liste personnes : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

@app.get("/person/{person_id}")
async def get_person(person_id: str):
    try:
        person = database.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Personne non trouvée")
        
        return {
            "success": True,
            "person": person
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération personne : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

@app.delete("/person/{person_id}")
async def delete_person(person_id: str):
    try:
        success = database.delete_person(person_id)
        if not success:
            raise HTTPException(status_code=404, detail="Personne non trouvée")
        
        logger.info(f"🗑️ Personne {person_id} supprimée")
        
        return {
            "success": True,
            "message": f"Personne {person_id} supprimée"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur suppression : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

@app.get("/stats")
async def get_stats():
    try:
        return {
            "success": True,
            "statistics": {
                "total_persons": len(database.persons),
                "face_engine": "OpenCV Haar Cascade",
                "database_type": "Memory",
                "version": "0.1.0-test"
            }
        }
    except Exception as e:
        logger.error(f"❌ Erreur stats : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

# Point d'entrée
if __name__ == "__main__":
    logger.info("🚀 Démarrage de l'API de test...")
    logger.info(f"🌐 API : http://localhost:{config.api_port}")
    logger.info(f"📖 Documentation : http://localhost:{config.api_port}/docs")
    
    # Créer les dossiers nécessaires
    folders = ["storage/uploads", "storage/database", "storage/temp"]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info"
    )
