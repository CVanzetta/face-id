# Face-ID Recognition System

Système de reconnaissance faciale basé sur InsightFace (RetinaFace + ArcFace) avec indexation FAISS et tracking vidéo.

## Stack Technique

- **InsightFace** : Détection faciale (RetinaFace) + Embeddings (ArcFace)
- **FAISS** : Recherche vectorielle rapide
- **OpenCV** : Traitement vidéo et image
- **Tracker simple** : Stabilisation d'ID en vidéo

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration GPU

Si vous n'avez pas de GPU NVIDIA, remplacez `onnxruntime-gpu` par `onnxruntime` dans `requirements.txt`.

## Utilisation

### 1. Préparation des données

Placez 3-10 photos par personne dans `data/people/<prenom_nom>/` avec des éclairages et angles variés.

```
data/people/
├─ alice/
│  ├─ img1.jpg
│  ├─ img2.jpg
│  └─ img3.jpg
└─ bob/
   ├─ img1.jpg
   └─ img2.jpg
```

### 2. Indexation

```bash
python scripts/index_people.py
```

Cela crée `data/outputs/index.faiss` et `labels.json`.

### 3. Test sur photo

```bash
python scripts/identify_image.py --img path/vers/photo.jpg
```

### 4. Test en vidéo (webcam)

```bash
python scripts/identify_video.py --video 0
```

Pour un fichier vidéo :
```bash
python scripts/identify_video.py --video path/clip.mp4
```

## Configuration

Éditez `config.yaml` pour ajuster :
- `threshold_cosine` : Seuil de reconnaissance (0.30-0.45)
- `device` : 0 pour GPU, -1 pour CPU
- `video.detect_every_n_frames` : Fréquence de détection
- `video.track_max_age` : Durée de vie des tracks
- `video.smooth_window` : Fenêtre de lissage

## Calibration

Pour optimiser la précision, testez différentes valeurs de `threshold_cosine` avec vos données de validation.

## RGPD

Assurez-vous d'avoir le consentement explicite des personnes avant d'utiliser leurs photos.

## Structure du projet

```
face-id/
├─ README.md
├─ requirements.txt
├─ config.yaml
├─ data/
│  ├─ people/                    # Photos d'apprentissage
│  └─ outputs/                   # Index FAISS et labels
├─ scripts/
│  ├─ index_people.py            # Création de l'index
│  ├─ identify_image.py          # Test sur photo
│  ├─ identify_video.py          # Test sur vidéo
│  └─ utils/
│     ├─ face_engine.py          # Moteur de détection/embedding
│     ├─ index_store.py          # Gestion index FAISS
│     └─ tracking.py             # Tracking vidéo
```
