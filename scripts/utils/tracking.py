from collections import deque
import numpy as np

class SimpleTrack:
    """Track = historique bbox + embeddings + scores pour smoothing."""
    def __init__(self, tid, bbox, emb, score, smooth_window=7):
        self.id = tid
        self.bbox = bbox
        self.embs = deque([emb], maxlen=smooth_window)
        self.scores = deque([score], maxlen=smooth_window)
        self.name = None
        self.name_score = 0.0
        self.misses = 0

    def update(self, bbox, emb, score):
        self.bbox = bbox
        self.embs.append(emb)
        self.scores.append(score)
        self.misses = 0

    def avg_embedding(self):
        return np.mean(np.stack(self.embs), axis=0)

    def avg_score(self):
        return float(np.mean(self.scores))

class Tracker:
    """Associe par IoU + met à jour les tracks (POC)."""
    def __init__(self, iou_thresh=0.3, max_age=15, smooth_window=7):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.smooth_window = smooth_window
        self.next_id = 1
        self.tracks = []

    @staticmethod
    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update(self, detections):
        # detections: list of dicts {bbox, embedding, score}
        assigned = set()
        # assign by IoU
        for det in detections:
            bbox = det["bbox"]
            best, best_iou, best_idx = None, 0.0, -1
            for i, tr in enumerate(self.tracks):
                iou = self.iou(tr.bbox, bbox)
                if iou > best_iou:
                    best, best_iou, best_idx = tr, iou, i
            if best and best_iou >= self.iou_thresh:
                self.tracks[best_idx].update(bbox, det["embedding"], det["score"])
                assigned.add(best_idx)
            else:
                # new track
                t = SimpleTrack(self.next_id, bbox, det["embedding"], det["score"], self.smooth_window)
                self.tracks.append(t)
                self.next_id += 1

        # aging
        for i, tr in enumerate(self.tracks):
            if i not in assigned:
                tr.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return self.tracks
