import argparse, yaml, cv2, numpy as np, json
from utils.face_engine import FaceEngine
from utils.index_store import IndexStore

def load_config():
    return yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--index", default="data/outputs/index.faiss")
    ap.add_argument("--labels", default="data/outputs/labels.json")
    args = ap.parse_args()

    cfg = load_config()
    fe = FaceEngine(device=cfg["device"], det_size=tuple(cfg["det_size"]))
    idx = IndexStore(); idx.load(args.index, args.labels)

    img = cv2.imread(args.img)
    faces = fe.detect_and_embed(img)
    for f in faces:
        q = f["embedding"].reshape(1, -1)
        D, I = idx.search(q, k=cfg["top_k"])
        score = float(D[0,0])
        name = idx.labels[I[0,0]] if score >= cfg["threshold_cosine"] else "Inconnu"
        x1,y1,x2,y2 = f["bbox"]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{name} ({score:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        print(name, score)
    cv2.imshow("result", img); cv2.waitKey(0)
