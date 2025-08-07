"""
Enhanced Face Recognition Engine with anti-spoofing and quality assessment
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import time

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
except ImportError:
    raise ImportError("InsightFace not installed. Run: pip install insightface")

logger = logging.getLogger(__name__)

class FaceRecognitionEngine:
    """
    Advanced face recognition engine with quality assessment and anti-spoofing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.face_config = config.get('face_recognition', {})
        self.thresholds = config.get('thresholds', {})
        
        # Initialize InsightFace
        self.app = None
        self.detector = None
        self.recognizer = None
        self._load_models()
        
        logger.info(f"Face engine initialized with device: {self.face_config.get('device', 0)}")
    
    def _load_models(self):
        """Load face detection and recognition models"""
        try:
            device_id = self.face_config.get('device', 0)
            det_size = tuple(self.face_config.get('det_size', [640, 640]))
            model_name = self.face_config.get('model_name', 'buffalo_l')
            
            # Load full pipeline
            self.app = FaceAnalysis(name=model_name)
            self.app.prepare(ctx_id=device_id, det_size=det_size)
            
            logger.info(f"Loaded InsightFace model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load face models: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray, return_landmarks: bool = True) -> List[Dict]:
        """
        Detect faces in image with quality assessment
        
        Args:
            image: Input image (BGR format)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            List of face detections with metadata
        """
        if self.app is None:
            raise RuntimeError("Face detection model not loaded")
        
        try:
            # Get faces from InsightFace
            faces = self.app.get(image)
            
            results = []
            for face in faces:
                # Basic detection info
                face_data = {
                    'bbox': face.bbox.astype(int).tolist(),
                    'confidence': float(face.det_score),
                    'embedding': face.normed_embedding.astype(np.float32),
                }
                
                # Add landmarks if requested
                if return_landmarks and hasattr(face, 'kps'):
                    face_data['landmarks'] = face.kps.flatten().tolist()
                
                # Face quality assessment
                face_data['quality_score'] = self._assess_face_quality(image, face)
                
                # Basic anti-spoofing (can be enhanced with dedicated models)
                face_data['liveness_score'] = self._basic_liveness_check(image, face)
                
                # Age and gender (if available)
                if hasattr(face, 'age'):
                    face_data['age'] = int(face.age)
                if hasattr(face, 'gender'):
                    face_data['gender'] = int(face.gender)
                
                results.append(face_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_embedding(self, image: np.ndarray, bbox: Optional[List] = None) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image: Input image
            bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Face embedding vector or None
        """
        faces = self.detect_faces(image, return_landmarks=False)
        
        if not faces:
            return None
        
        # If bbox provided, find closest face
        if bbox is not None:
            best_face = self._find_closest_face(faces, bbox)
        else:
            # Take highest confidence face
            best_face = max(faces, key=lambda x: x['confidence'])
        
        return best_face.get('embedding')
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure embeddings are normalized
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def verify_faces(self, img1: np.ndarray, img2: np.ndarray, 
                    threshold: Optional[float] = None) -> Dict:
        """
        Verify if two images contain the same person
        
        Args:
            img1: First image
            img2: Second image
            threshold: Custom similarity threshold
            
        Returns:
            Verification result with metadata
        """
        if threshold is None:
            threshold = self.thresholds.get('cosine_similarity', 0.35)
        
        start_time = time.time()
        
        # Extract embeddings
        faces1 = self.detect_faces(img1)
        faces2 = self.detect_faces(img2)
        
        result = {
            'is_match': False,
            'similarity_score': 0.0,
            'confidence': 0.0,
            'faces_detected': {'image1': len(faces1), 'image2': len(faces2)},
            'processing_time': 0.0,
            'threshold_used': threshold
        }
        
        if not faces1 or not faces2:
            result['processing_time'] = time.time() - start_time
            return result
        
        # Compare best faces (highest confidence)
        best_face1 = max(faces1, key=lambda x: x['confidence'])
        best_face2 = max(faces2, key=lambda x: x['confidence'])
        
        similarity = self.compare_embeddings(
            best_face1['embedding'], 
            best_face2['embedding']
        )
        
        result.update({
            'similarity_score': similarity,
            'is_match': similarity >= threshold,
            'confidence': min(best_face1['confidence'], best_face2['confidence']),
            'processing_time': time.time() - start_time
        })
        
        return result
    
    def _assess_face_quality(self, image: np.ndarray, face) -> float:
        """
        Assess face quality based on various factors
        
        Args:
            image: Input image
            face: Face detection result
            
        Returns:
            Quality score (0-1)
        """
        try:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return 0.0
            
            # Face size factor (larger faces generally better)
            face_size = (x2 - x1) * (y2 - y1)
            img_size = image.shape[0] * image.shape[1]
            size_factor = min(1.0, face_size / (img_size * 0.01))  # 1% of image is good
            
            # Sharpness assessment using Laplacian variance
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_factor = min(1.0, laplacian_var / 100.0)  # Normalize
            
            # Brightness assessment
            brightness = np.mean(gray_face)
            brightness_factor = 1.0 - abs(brightness - 128) / 128.0  # Optimal around 128
            
            # Detection confidence
            conf_factor = face.det_score
            
            # Combined quality score
            quality = (size_factor * 0.3 + 
                      sharpness_factor * 0.3 + 
                      brightness_factor * 0.2 + 
                      conf_factor * 0.2)
            
            return min(1.0, quality)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default moderate quality
    
    def _basic_liveness_check(self, image: np.ndarray, face) -> float:
        """
        Basic anti-spoofing check (placeholder for more advanced methods)
        
        Args:
            image: Input image
            face: Face detection result
            
        Returns:
            Liveness score (0-1)
        """
        try:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return 0.0
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            
            # Check for color diversity (real faces have more color variation)
            color_diversity = np.std(hsv[:, :, 1])  # Saturation variation
            diversity_score = min(1.0, color_diversity / 50.0)
            
            # Texture analysis (real skin has micro-textures)
            texture_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0)
            
            # Simple reflection check (photos often have uniform lighting)
            lighting_variance = np.std(gray)
            lighting_score = min(1.0, lighting_variance / 50.0)
            
            # Combined liveness score
            liveness = (diversity_score * 0.4 + 
                       texture_score * 0.4 + 
                       lighting_score * 0.2)
            
            return min(1.0, liveness)
            
        except Exception as e:
            logger.warning(f"Liveness check failed: {e}")
            return 0.7  # Default moderate liveness
    
    def _find_closest_face(self, faces: List[Dict], target_bbox: List) -> Dict:
        """Find face closest to target bounding box"""
        def bbox_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            inter = max(0, x2-x1) * max(0, y2-y1)
            area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
            area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
            union = area1 + area2 - inter + 1e-6
            
            return inter / union
        
        best_face = faces[0]
        best_iou = 0.0
        
        for face in faces:
            iou = bbox_iou(face['bbox'], target_bbox)
            if iou > best_iou:
                best_iou = iou
                best_face = face
        
        return best_face
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'model_name': self.face_config.get('model_name', 'buffalo_l'),
            'device': self.face_config.get('device', 0),
            'det_size': self.face_config.get('det_size', [640, 640]),
            'loaded': self.app is not None,
            'thresholds': self.thresholds
        }
