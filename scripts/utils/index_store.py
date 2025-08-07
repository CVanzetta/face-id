import json, faiss, numpy as np
from pathlib import Path

class IndexStore:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine ~ inner product si vecteurs L2-normés
        self.labels = []

    def add(self, vecs: np.ndarray, labels):
        assert vecs.dtype == np.float32
        self.index.add(vecs)
        self.labels.extend(labels)

    def search(self, q: np.ndarray, k: int = 1):
        D, I = self.index.search(q, k)
        return D, I

    def save(self, index_path: str, labels_path: str):
        faiss.write_index(self.index, index_path)
        Path(labels_path).write_text(json.dumps(self.labels, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, index_path: str, labels_path: str):
        self.index = faiss.read_index(index_path)
        self.labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.dim = self.index.d
