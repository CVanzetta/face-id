"""
Moteur de reconnaissance faciale temporaire utilisant OpenCV
Version de démarrage sans InsightFace
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import requests
import os

logger = logging.getLogger(__name__)

class FaceRecognitionEngine:
    """Moteur de reconnaissance faciale utilisant OpenCV"""
    
    def __init__(self, config):
        """Initialise le moteur de reconnaissance faciale"""
        self.config = config
        self.face_cascade = None
        self.is_initialized = False
        
        # Initialiser le détecteur de visages OpenCV
        self._init_opencv_detector()
        
    def _init_opencv_detector(self):
        """Initialise le détecteur de visages OpenCV"""
        try:
            # Utiliser le détecteur par défaut d'OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Impossible de charger le modèle Haar Cascade")
            
            logger.info("✅ Détecteur de visages OpenCV initialisé")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du détecteur : {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Détecte les visages dans une image"""
        if not self.is_initialized:
            logger.error("❌ Moteur non initialisé")
            return []
        
        try:
            # Convertir en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Détecter les visages
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convertir en format attendu
            detected_faces = []
            for i, (x, y, w, h) in enumerate(faces):
                # Extraire la région du visage
                face_region = image[y:y+h, x:x+w]
                
                # Générer un embedding simple basé sur les caractéristiques du visage
                embedding = self._generate_simple_embedding(face_region)
                
                face_data = {
                    'bbox': [x, y, x+w, y+h],
                    'landmarks': self._estimate_landmarks(x, y, w, h),
                    'embedding': embedding,
                    'confidence': 0.8,  # Confiance fixe pour OpenCV
                    'det_score': 0.8
                }
                detected_faces.append(face_data)
            
            logger.debug(f"🔍 Détecté {len(detected_faces)} visage(s)")
            return detected_faces
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la détection : {e}")
            return []
    
    def _generate_simple_embedding(self, face_region: np.ndarray) -> np.ndarray:
        """Génère un embedding simple basé sur les caractéristiques du visage"""
        try:
            # Redimensionner le visage à une taille fixe
            face_resized = cv2.resize(face_region, (64, 64))
            
            # Convertir en niveaux de gris si nécessaire
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Normaliser
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Calculer des caractéristiques simples
            # 1. Histogramme des pixels
            hist = cv2.calcHist([face_gray], [0], None, [16], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-7)
            
            # 2. Caractéristiques LBP simplifiées
            lbp_features = self._calculate_lbp_features(face_gray)
            
            # 3. Moyennes par régions
            h, w = face_gray.shape
            regions = [
                face_gray[0:h//3, 0:w//3],      # Top-left
                face_gray[0:h//3, w//3:2*w//3], # Top-center
                face_gray[0:h//3, 2*w//3:w],    # Top-right
                face_gray[h//3:2*h//3, 0:w//3], # Mid-left
                face_gray[h//3:2*h//3, w//3:2*w//3], # Center
                face_gray[h//3:2*h//3, 2*w//3:w],    # Mid-right
                face_gray[2*h//3:h, 0:w//3],    # Bottom-left
                face_gray[2*h//3:h, w//3:2*w//3], # Bottom-center
                face_gray[2*h//3:h, 2*w//3:w],    # Bottom-right
            ]
            regional_means = [np.mean(region) for region in regions]
            
            # Combiner toutes les caractéristiques
            embedding = np.concatenate([
                hist,
                lbp_features,
                regional_means
            ])
            
            # Normaliser l'embedding final
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de l'embedding : {e}")
            # Retourner un embedding aléatoire en cas d'erreur
            return np.random.rand(128).astype(np.float32)
    
    def _calculate_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Calcule des caractéristiques LBP simplifiées"""
        try:
            h, w = image.shape
            if h < 3 or w < 3:
                return np.zeros(16)
                
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = image[i, j]
                    binary_code = 0
                    
                    # Comparer avec les 8 pixels voisins
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            binary_code |= (1 << k)
                    
                    lbp[i-1, j-1] = binary_code % 256
            
            # Calculer l'histogramme LBP
            hist_lbp = cv2.calcHist([lbp], [0], None, [16], [0, 256])
            hist_normalized = hist_lbp.flatten() / (np.sum(hist_lbp) + 1e-7)
            return hist_normalized
            
        except Exception:
            return np.zeros(16)
    
    def _estimate_landmarks(self, x: int, y: int, w: int, h: int) -> List[List[float]]:
        """Estime des landmarks de base pour un visage détecté"""
        # Points clés estimés pour un visage rectangulaire
        landmarks = [
            [x + w*0.3, y + h*0.4],  # Œil gauche
            [x + w*0.7, y + h*0.4],  # Œil droit
            [x + w*0.5, y + h*0.6],  # Nez
            [x + w*0.3, y + h*0.8],  # Coin gauche de la bouche
            [x + w*0.7, y + h*0.8],  # Coin droit de la bouche
        ]
        return landmarks
    
    def verify_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[bool, float]:
        """Vérifie si deux embeddings correspondent au même visage"""
        try:
            # Calculer la similarité cosinus
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return False, 0.0
                
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Seuil de vérification
            threshold = getattr(self.config.thresholds, 'cosine_similarity', 0.35)
            is_same = similarity > threshold
            
            return is_same, float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification : {e}")
            return False, 0.0
    
    def assess_face_quality(self, image: np.ndarray, face_data: Dict[str, Any]) -> float:
        """Évalue la qualité d'un visage détecté"""
        try:
            bbox = face_data['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extraire la région du visage
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return 0.0
            
            # Convertir en niveaux de gris
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region
            
            # Critères de qualité
            scores = []
            
            # 1. Taille du visage (plus grand = mieux)
            face_area = (x2 - x1) * (y2 - y1)
            size_score = min(1.0, face_area / (100 * 100))  # Normaliser par rapport à 100x100
            scores.append(size_score)
            
            # 2. Netteté (variance de Laplacian)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normaliser
            scores.append(sharpness_score)
            
            # 3. Contraste
            contrast = gray_face.std()
            contrast_score = min(1.0, contrast / 64.0)  # Normaliser
            scores.append(contrast_score)
            
            # 4. Éclairage (pas trop sombre ou trop clair)
            mean_brightness = gray_face.mean()
            lighting_score = 1.0 - abs(mean_brightness - 128) / 128.0
            scores.append(lighting_score)
            
            # Score final (moyenne pondérée)
            weights = [0.2, 0.3, 0.3, 0.2]
            quality_score = sum(s * w for s, w in zip(scores, weights))
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'évaluation de qualité : {e}")
            return 0.5  # Score neutre en cas d'erreur
    
    def detect_liveness(self, image: np.ndarray, face_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Détection basique de vivacité (anti-spoofing)"""
        try:
            bbox = face_data['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extraire la région du visage
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return False, 0.0
            
            # Critères de vivacité simples
            scores = []
            
            # 1. Variation de texture (les photos imprimées ont moins de variation)
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region
            
            texture_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            texture_score = min(1.0, texture_var / 500.0)
            scores.append(texture_score)
            
            # 2. Variation de couleur (les écrans ont des couleurs différentes)
            if len(face_region.shape) == 3:
                color_var = np.var(face_region, axis=(0, 1))
                color_score = min(1.0, np.mean(color_var) / 1000.0)
                scores.append(color_score)
            else:
                scores.append(0.5)  # Score neutre pour les images en N&B
            
            # 3. Détection de bords (les photos imprimées ont des bords artificiels)
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = 1.0 - min(1.0, edge_density * 3)  # Moins de bords = plus vivant
            scores.append(edge_score)
            
            # Score de vivacité final
            liveness_score = np.mean(scores)
            threshold = getattr(self.config.thresholds, 'liveness_score', 0.7)
            is_live = liveness_score > threshold
            
            return is_live, float(liveness_score)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la détection de vivacité : {e}")
            return True, 0.5  # Supposer vivant en cas d'erreur
