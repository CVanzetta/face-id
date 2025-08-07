# 🚀 Démarrage Rapide - Système de Reconnaissance Faciale

## 📋 Prérequis

- Python 3.8+
- Webcam (optionnel pour la détection en temps réel)
- 4GB RAM minimum (8GB recommandé)
- GPU NVIDIA optionnel mais recommandé

## ⚡ Installation Express

### 1. Télécharger le projet
```bash
# Depuis le dossier actuel
cd "c:\Users\User\Desktop\Test IA\face-id"
```

### 2. Installation automatique
```bash
python quick_start.py
```

Cela va :
- ✅ Créer l'environnement virtuel Python
- ✅ Installer toutes les dépendances
- ✅ Télécharger les modèles InsightFace
- ✅ Optimiser la configuration pour votre système
- ✅ Créer les dossiers nécessaires

### 3. Démarrer l'API
```bash
python main.py
```

### 4. Accéder à l'interface web
Ouvrez votre navigateur et allez sur : http://localhost:8000/web

## 🎯 Premier Test

### Test Rapide
1. Allez sur l'onglet **"Enroll"** dans l'interface web
2. Uploadez 2-3 photos d'une personne
3. Donnez un nom (ex: "John Doe")
4. Cliquez sur "Enroll Person"
5. Allez sur l'onglet **"Search"**
6. Uploadez une autre photo de la même personne
7. Cliquez sur "Search" - vous devriez voir la correspondance !

### Test avec la Webcam
1. Allez sur l'onglet **"Live Search"**
2. Autoriser l'accès à la caméra
3. Positionnez-vous devant la webcam
4. Le système devrait détecter votre visage en temps réel

## 🔧 Configuration Personnalisée

Le fichier `config.yaml` contient tous les paramètres :

```yaml
# Seuil de reconnaissance (plus bas = plus permissif)
thresholds:
  cosine_similarity: 0.35  # 0.3-0.4 recommandé

# Performance GPU
face_recognition:
  device: 0  # 0 = GPU, -1 = CPU

# Qualité des images
quality:
  min_face_size: 50    # Taille minimum du visage en pixels
  blur_threshold: 50   # Netteté minimum
```

## 📊 Endpoints API Principaux

### Enregistrer une personne
```bash
curl -X POST "http://localhost:8000/enroll" \
  -F "name=Jean Dupont" \
  -F "source=upload" \
  -F "consent_type=explicit" \
  -F "files=@photo1.jpg"
```

### Rechercher un visage
```bash
curl -X POST "http://localhost:8000/search" \
  -F "file=@recherche.jpg" \
  -F "top_k=5" \
  -F "threshold=0.35"
```

### Vérifier deux visages
```bash
curl -X POST "http://localhost:8000/verify" \
  -F "file1=@image1.jpg" \
  -F "file2=@image2.jpg"
```

## 🐳 Alternative Docker

Si vous préférez utiliser Docker :

```bash
# Construire et lancer
docker-compose up --build

# Avec base de données Qdrant
docker-compose --profile qdrant up --build
```

## 🔍 Résolution de Problèmes

### GPU non détecté
```bash
# Vérifier CUDA
nvidia-smi

# Installer version GPU
pip install onnxruntime-gpu faiss-gpu
```

### Erreur de mémoire
- Réduire la taille des images d'entrée
- Utiliser le CPU au lieu du GPU
- Réduire le batch_size dans config.yaml

### Performance lente
- Activer le GPU dans config.yaml
- Précharger les modèles : `preload_models: true`
- Utiliser des images plus petites

## 📁 Structure des Données

Pour une organisation optimale :

```
storage/
├── uploads/           # Images uploadées
├── database/          # Base de données FAISS
├── temp/             # Fichiers temporaires
└── logs/             # Journaux système
```

## 🛡️ Conformité RGPD

Le système inclut :
- ✅ Gestion du consentement
- ✅ Droit à l'effacement (DELETE /person/{id})
- ✅ Minimisation des données
- ✅ Journalisation des accès
- ✅ Anonymisation des logs

## 📞 Support

- 📖 Documentation complète : http://localhost:8000/docs
- 🔧 Configuration : Éditer `config.yaml`
- 🧪 Tests : `python test_api.py`
- 📊 Statistiques : http://localhost:8000/stats

## 🎯 Cas d'Usage Recommandés

1. **Contrôle d'accès** : Authentification biométrique
2. **Recherche de personnes** : Dans bases de données sécurisées
3. **Présence/pointage** : Suivi automatique
4. **Sécurité événementielle** : Identification en temps réel

⚠️ **Important** : Toujours obtenir un consentement explicite avant d'utiliser des données biométriques !

---

🚀 **Prêt à démarrer ?** Exécutez `python main.py` et allez sur http://localhost:8000/web
