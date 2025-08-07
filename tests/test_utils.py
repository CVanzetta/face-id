import unittest
import numpy as np
import sys
from pathlib import Path

# Ajoute le dossier scripts au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils.index_store import IndexStore
from utils.tracking import SimpleTrack, Tracker

class TestIndexStore(unittest.TestCase):
    def setUp(self):
        self.store = IndexStore(dim=128)
    
    def test_add_and_search(self):
        # Données de test
        vectors = np.random.randn(3, 128).astype("float32")
        labels = ["alice", "bob", "charlie"]
        
        # Ajout
        self.store.add(vectors, labels)
        
        # Test de recherche
        query = vectors[0:1]  # Premier vecteur
        D, I = self.store.search(query, k=1)
        
        # Vérifications
        self.assertEqual(len(D), 1)
        self.assertEqual(len(I), 1)
        self.assertEqual(I[0, 0], 0)  # Doit retrouver le premier vecteur
        self.assertEqual(self.store.labels[I[0, 0]], "alice")

class TestTracking(unittest.TestCase):
    def test_simple_track(self):
        bbox = [10, 10, 50, 50]
        emb = np.random.randn(128).astype("float32")
        score = 0.8
        
        track = SimpleTrack(1, bbox, emb, score, smooth_window=3)
        
        # Test initial
        self.assertEqual(track.id, 1)
        self.assertEqual(len(track.embs), 1)
        self.assertEqual(track.avg_score(), 0.8)
        
        # Test mise à jour
        new_bbox = [12, 12, 52, 52]
        new_emb = np.random.randn(128).astype("float32")
        new_score = 0.9
        
        track.update(new_bbox, new_emb, new_score)
        
        self.assertEqual(len(track.embs), 2)
        self.assertEqual(track.avg_score(), 0.85)
        self.assertEqual(track.misses, 0)
    
    def test_tracker_iou(self):
        # Test de calcul IoU
        bbox1 = [0, 0, 10, 10]  # 100 pixels
        bbox2 = [5, 5, 15, 15]  # 100 pixels, intersection 25 pixels
        
        iou = Tracker.iou(bbox1, bbox2)
        expected_iou = 25 / (100 + 100 - 25)  # intersection / union
        
        self.assertAlmostEqual(iou, expected_iou, places=5)

if __name__ == "__main__":
    unittest.main()
