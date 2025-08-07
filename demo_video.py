#!/usr/bin/env python3
"""
Script de démonstration vidéo du système de reconnaissance faciale
Lance une démo en temps réel avec la webcam
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

class FaceRecognitionDemo:
    """Classe pour la démonstration du système de reconnaissance faciale"""
    
    def __init__(self):
        """Initialise la démonstration"""
        self.config = Config()
        self.face_engine = None
        self.database = None
        self.known_faces = {}
        
    async def initialize(self):
        """Initialise les composants"""
        try:
            logger.info("Initialisation du moteur de reconnaissance faciale...")
            self.face_engine = FaceRecognitionEngine(self.config)
            
            logger.info("Initialisation de la base de données...")
            self.database = create_database_service(self.config)
            
            # Charger les visages connus
            await self.load_known_faces()
            
            logger.info("Initialisation terminée ✅")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation : {e}")
            return False
    
    async def load_known_faces(self):
        """Charge les visages déjà enregistrés"""
        try:
            persons = await self.database.list_persons()
            self.known_faces = {person['id']: person['name'] for person in persons}
            logger.info(f"Chargé {len(self.known_faces)} personnes connues")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des visages : {e}")
    
    def draw_face_info(self, frame, face_data, x1, y1, x2, y2):
        """Dessine les informations du visage sur l'image"""
        # Rectangle autour du visage
        color = (0, 255, 0) if face_data['match'] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Informations textuelles
        label_lines = []
        
        if face_data['match']:
            label_lines.append(f"ID: {face_data['name']}")
            label_lines.append(f"Conf: {face_data['confidence']:.2f}")
        else:
            label_lines.append("Inconnu")
            label_lines.append(f"Conf: {face_data['confidence']:.2f}")
        
        # Qualité du visage
        quality_score = face_data.get('quality', 0.0)
        label_lines.append(f"Qualite: {quality_score:.2f}")
        
        # Afficher les lignes de texte
        for i, line in enumerate(label_lines):
            y_text = y1 - 10 - (len(label_lines) - 1 - i) * 25
            if y_text < 0:
                y_text = y2 + 25 + i * 25
            
            # Fond pour le texte
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, 
                (x1, y_text - text_height - baseline),
                (x1 + text_width, y_text + baseline),
                color, 
                -1
            )
            
            # Texte
            cv2.putText(
                frame, line, (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
    
    async def process_frame(self, frame):
        """Traite une frame de la vidéo"""
        try:
            # Détecter les visages
            faces = self.face_engine.detect_faces(frame)
            
            results = []
            for face in faces:
                bbox = face['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Évaluer la qualité
                quality_score = self.face_engine.assess_face_quality(frame, face)
                
                # Rechercher dans la base de données
                search_results = await self.database.search_similar(
                    face['embedding'],
                    top_k=1,
                    threshold=self.config.thresholds.cosine_similarity
                )
                
                # Préparer les données d'affichage
                face_data = {
                    'bbox': bbox,
                    'quality': quality_score,
                    'match': False,
                    'name': 'Inconnu',
                    'confidence': 0.0
                }
                
                if search_results and len(search_results) > 0:
                    best_match = search_results[0]
                    face_data.update({
                        'match': True,
                        'name': best_match['person_name'],
                        'confidence': best_match['similarity']
                    })
                
                results.append(face_data)
                
                # Dessiner les informations
                self.draw_face_info(frame, face_data, x1, y1, x2, y2)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la frame : {e}")
            return []
    
    def add_stats_overlay(self, frame, stats):
        """Ajoute les statistiques sur l'image"""
        overlay_text = [
            f"FPS: {stats['fps']:.1f}",
            f"Visages detectes: {stats['faces_detected']}",
            f"Reconnus: {stats['faces_recognized']}",
            f"Total personnes: {len(self.known_faces)}",
            "",
            "Controles:",
            "ESPACE: Capturer",
            "Q: Quitter",
            "S: Statistiques"
        ]
        
        for i, line in enumerate(overlay_text):
            y_pos = 30 + i * 25
            cv2.putText(
                frame, line, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame, line, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                cv2.LINE_AA
            )
    
    async def run_demo(self):
        """Lance la démonstration"""
        if not await self.initialize():
            logger.error("Impossible d'initialiser la démonstration")
            return
        
        # Ouvrir la webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Impossible d'ouvrir la webcam")
            return
        
        # Configuration de la webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Démonstration démarrée - Appuyez sur 'Q' pour quitter")
        
        # Variables pour les statistiques
        frame_count = 0
        start_time = time.time()
        total_faces_detected = 0
        total_faces_recognized = 0
        results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Impossible de lire la frame de la webcam")
                    break
                
                frame_count += 1
                
                # Traiter la frame (pas à chaque frame pour optimiser)
                if frame_count % 3 == 0:  # Traiter 1 frame sur 3
                    results = await self.process_frame(frame)
                    total_faces_detected += len(results)
                    total_faces_recognized += sum(1 for r in results if r['match'])
                
                # Calculer les statistiques
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                stats = {
                    'fps': fps,
                    'faces_detected': len(results),
                    'faces_recognized': sum(1 for r in results if r['match']),
                    'total_faces_detected': total_faces_detected,
                    'total_faces_recognized': total_faces_recognized
                }
                
                # Ajouter l'overlay d'informations
                self.add_stats_overlay(frame, stats)
                
                # Afficher la frame
                cv2.imshow('Face Recognition Demo', frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord(' '):  # Espace pour capturer
                    timestamp = int(time.time())
                    filename = f"capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Image sauvegardée : {filename}")
                elif key == ord('s') or key == ord('S'):  # Statistiques
                    logger.info("Statistiques:")
                    logger.info(f"   - FPS moyen: {fps:.1f}")
                    logger.info(f"   - Total visages détectés: {total_faces_detected}")
                    logger.info(f"   - Total visages reconnus: {total_faces_recognized}")
                    if total_faces_detected > 0:
                        rate = (total_faces_recognized/total_faces_detected*100)
                        logger.info(f"   - Taux de reconnaissance: {rate:.1f}%")
        
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Démonstration terminée")

async def main():
    """Fonction principale"""
    print("🚀 Démonstration du Système de Reconnaissance Faciale")
    print("=" * 60)
    print()
    print("Cette démonstration va :")
    print("• Détecter les visages en temps réel via votre webcam")
    print("• Reconnaître les personnes déjà enregistrées")
    print("• Afficher la qualité et la confiance des détections")
    print("• Fournir des statistiques en temps réel")
    print()
    print("Assurez-vous d'avoir des personnes enregistrées dans le système !")
    print("Utilisez l'interface web (http://localhost:8000/web) pour enregistrer des visages.")
    print()
    
    # Demander confirmation
    try:
        input("Appuyez sur Entrée pour commencer la démonstration...")
    except KeyboardInterrupt:
        print("\nDémonstration annulée.")
        return
    
    # Lancer la démonstration
    demo = FaceRecognitionDemo()
    await demo.run_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Erreur fatale : {e}")
        sys.exit(1)
