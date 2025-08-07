import argparse, json, yaml, cv2, numpy as np
from pathlib import Path
from utils.face_engine import FaceEngine
from utils.index_store import IndexStore

def load_config():
    return yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

def iter_person_images(root):
    root = Path(root)
    for person_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        imgs = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
        if imgs:
            yield person_dir.name, imgs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--people_dir", default="data/people")
    ap.add_argument("--out_index", default="data/outputs/index.faiss")
    ap.add_argument("--out_labels", default="data/outputs/labels.json")
    args = ap.parse_args()

    cfg = load_config()
    fe = FaceEngine(device=cfg["device"], det_size=tuple(cfg["det_size"]))
    vectors, labels = [], []

    for person, img_paths in iter_person_images(args.people_dir):
        embs = []
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None: continue
            f = fe.best_face_embedding(img)
            if f is not None:
                embs.append(f["embedding"])
        if embs:
            vec = np.mean(np.stack(embs), axis=0).astype("float32")
            vectors.append(vec)
            labels.append(person)
            print(f"[OK] {person}: {len(embs)} faces")

    if not vectors:
        raise SystemExit("Aucune embedding générée.")

    xb = np.vstack(vectors)
    idx = IndexStore(dim=xb.shape[1])
    idx.add(xb, labels)
    Path(args.out_index).parent.mkdir(parents=True, exist_ok=True)
    idx.save(args.out_index, args.out_labels)
    print(f"Index sauvegardé → {args.out_index}\nLabels → {args.out_labels}")
