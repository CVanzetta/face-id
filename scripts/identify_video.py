import argparse, yaml, cv2, numpy as np
from utils.face_engine import FaceEngine
from utils.index_store import IndexStore
from utils.tracking import Tracker

def load_config():
    return yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=0)  # 0 = webcam
    ap.add_argument("--index", default="data/outputs/index.faiss")
    ap.add_argument("--labels", default="data/outputs/labels.json")
    args = ap.parse_args()

    cfg = load_config()
    fe = FaceEngine(device=cfg["device"], det_size=tuple(cfg["det_size"]))
    idx = IndexStore(); idx.load(args.index, args.labels)
    cap = cv2.VideoCapture(0 if str(args.video)=="0" else args.video)

    tracker = Tracker(max_age=cfg["video"]["track_max_age"], smooth_window=cfg["video"]["smooth_window"])
    fcount = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        detections = []
        # Détecte 1 fois toutes les N frames
        if fcount % cfg["video"]["detect_every_n_frames"] == 0:
            faces = fe.detect_and_embed(frame)
            for f in faces:
                detections.append({"bbox": f["bbox"], "embedding": f["embedding"], "score": f["score"]})
        tracks = tracker.update(detections)

        # Identification (moyenne embedding par track)
        for t in tracks:
            q = t.avg_embedding().reshape(1,-1).astype("float32")
            D, I = idx.search(q, 1)
            score = float(D[0,0])
            if score >= cfg["threshold_cosine"]:
                name = idx.labels[I[0,0]]
                t.name, t.name_score = name, score
            else:
                t.name, t.name_score = "Inconnu", score

            x1,y1,x2,y2 = map(int, t.bbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {t.id} | {t.name} ({t.name_score:.2f})", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Face-ID", frame)
        fcount += 1
        if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
            break

    cap.release(); cv2.destroyAllWindows()
