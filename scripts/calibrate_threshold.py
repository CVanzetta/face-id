import argparse, yaml, cv2, numpy as np, json
from pathlib import Path
from utils.face_engine import FaceEngine
from utils.index_store import IndexStore
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def load_config():
    return yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

def create_validation_pairs(people_dir, test_ratio=0.2):
    """Crée des paires de validation (même personne vs différentes personnes)"""
    people_dir = Path(people_dir)
    same_pairs = []
    diff_pairs = []
    
    people = {}
    for person_dir in people_dir.iterdir():
        if person_dir.is_dir():
            imgs = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
            if len(imgs) >= 2:
                people[person_dir.name] = imgs
    
    # Paires de la même personne
    for name, imgs in people.items():
        n_test = max(1, int(len(imgs) * test_ratio))
        test_imgs = imgs[-n_test:]
        train_imgs = imgs[:-n_test]
        
        for i, img1 in enumerate(test_imgs):
            for img2 in train_imgs:
                same_pairs.append((img1, img2, 1, name))
    
    # Paires de personnes différentes
    names = list(people.keys())
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            imgs1 = people[name1][-max(1, int(len(people[name1]) * test_ratio)):]
            imgs2 = people[name2][-max(1, int(len(people[name2]) * test_ratio)):]
            
            for img1 in imgs1[:2]:  # Limite pour éviter trop de paires
                for img2 in imgs2[:2]:
                    diff_pairs.append((img1, img2, 0, f"{name1}_vs_{name2}"))
    
    return same_pairs + diff_pairs

def compute_similarity_scores(pairs, face_engine):
    """Calcule les scores de similarité pour toutes les paires"""
    scores = []
    labels = []
    
    for img1_path, img2_path, label, pair_name in pairs:
        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                continue
                
            face1 = face_engine.best_face_embedding(img1)
            face2 = face_engine.best_face_embedding(img2)
            
            if face1 is None or face2 is None:
                continue
                
            # Calcul de la similarité cosine
            emb1 = face1["embedding"]
            emb2 = face2["embedding"]
            similarity = np.dot(emb1, emb2)  # Les embeddings sont déjà normalisés
            
            scores.append(similarity)
            labels.append(label)
            
            print(f"{'✓' if label else '✗'} {pair_name}: {similarity:.3f}")
            
        except Exception as e:
            print(f"Erreur pour {pair_name}: {e}")
            continue
    
    return np.array(scores), np.array(labels)

def find_optimal_threshold(scores, labels):
    """Trouve le seuil optimal basé sur la courbe ROC"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Trouve le seuil qui maximise (TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, roc_auc, fpr, tpr, thresholds

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibration du seuil de reconnaissance")
    ap.add_argument("--people_dir", default="data/people")
    ap.add_argument("--output", default="data/outputs/calib_metrics.json")
    ap.add_argument("--plot", action="store_true", help="Affiche la courbe ROC")
    args = ap.parse_args()

    cfg = load_config()
    fe = FaceEngine(device=cfg["device"], det_size=tuple(cfg["det_size"]))
    
    print("Création des paires de validation...")
    pairs = create_validation_pairs(args.people_dir)
    print(f"Créé {len(pairs)} paires de validation")
    
    print("\nCalcul des scores de similarité...")
    scores, labels = compute_similarity_scores(pairs, fe)
    
    if len(scores) == 0:
        print("Aucun score calculé. Vérifiez vos données.")
        exit(1)
    
    print(f"\nAnalyse de {len(scores)} paires:")
    print(f"- Même personne: {np.sum(labels)} paires")
    print(f"- Personnes différentes: {np.sum(1-labels)} paires")
    
    # Calcul du seuil optimal
    optimal_thresh, roc_auc, fpr, tpr, thresholds = find_optimal_threshold(scores, labels)
    
    # Calcul des métriques pour différents seuils
    test_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, optimal_thresh]
    results = {}
    
    for thresh in test_thresholds:
        predictions = scores >= thresh
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)
        
        results[f"threshold_{thresh:.3f}"] = {
            "threshold": float(thresh),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }
    
    # Résultats
    print(f"\n=== RÉSULTATS DE CALIBRATION ===")
    print(f"Seuil optimal (ROC): {optimal_thresh:.3f}")
    print(f"AUC: {roc_auc:.3f}")
    print(f"\nComparaison des seuils:")
    print(f"{'Seuil':<8} {'Précision':<10} {'Rappel':<8} {'F1':<8} {'Précision':<10}")
    
    for thresh_key, metrics in results.items():
        t = metrics['threshold']
        p = metrics['precision']
        r = metrics['recall']
        f1 = metrics['f1_score']
        acc = metrics['accuracy']
        print(f"{t:<8.3f} {p:<10.3f} {r:<8.3f} {f1:<8.3f} {acc:<10.3f}")
    
    # Sauvegarde
    calib_data = {
        "optimal_threshold": float(optimal_thresh),
        "roc_auc": float(roc_auc),
        "current_config_threshold": float(cfg["threshold_cosine"]),
        "metrics_by_threshold": results,
        "recommendation": f"Utiliser seuil {optimal_thresh:.3f} pour un bon équilibre précision/rappel"
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nMétriques sauvegardées dans {args.output}")
    print(f"Recommandation: Modifier threshold_cosine à {optimal_thresh:.3f} dans config.yaml")
    
    # Affichage de la courbe ROC si demandé
    if args.plot:
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        except ImportError:
            print("matplotlib non installé, impossible d'afficher la courbe ROC")
