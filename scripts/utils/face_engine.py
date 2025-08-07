import cv2
from typing import List, Tuple
import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, device: int = 0, det_size=(640,640)):
        self.app = FaceAnalysis(name='buffalo_l')  # RetinaFace + ArcFace
        self.app.prepare(ctx_id=device, det_size=det_size)

    def detect_and_embed(self, img: np.ndarray):
        """Retourne une liste de dicts: {bbox, kps, score, embedding}"""
        faces = self.app.get(img)
        out = []
        for f in faces:
            out.append({
                "bbox": f.bbox.astype(int),
                "kps": f.kps,
                "score": float(f.det_score),
                "embedding": f.normed_embedding.astype("float32"),
            })
        return out

    def best_face_embedding(self, img: np.ndarray):
        faces = self.detect_and_embed(img)
        if not faces:
            return None
        f = max(faces, key=lambda x: x["score"])
        return f
