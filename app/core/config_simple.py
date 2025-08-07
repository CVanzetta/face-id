"""
Configuration simplifiée pour l'application
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration de l'application"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Configuration par défaut
        self.default_config = {
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'max_upload_size': 50,
                'cors_origins': ['*']
            },
            'face_recognition': {
                'device': -1,  # CPU par défaut
                'det_size': [640, 640],
                'model_name': 'buffalo_l'
            },
            'thresholds': {
                'cosine_similarity': 0.35,
                'liveness_score': 0.7,
                'face_quality': 0.5
            },
            'database': {
                'type': 'faiss',
                'faiss': {
                    'index_path': 'storage/database/face_index.faiss',
                    'metadata_path': 'storage/database/metadata.json'
                }
            },
            'security': {
                'consent_required': True,
                'data_retention_days': 365,
                'anonymize_logs': True
            }
        }
        
        # Charger la configuration depuis le fichier si il existe
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    self._merge_config(self.default_config, file_config)
            except Exception as e:
                print(f"Erreur lors du chargement de la configuration : {e}")
                print("Utilisation de la configuration par défaut")
        
        # Convertir en attributs d'objet
        self._dict_to_obj(self.default_config)
    
    def _merge_config(self, default: Dict[str, Any], file_config: Dict[str, Any]):
        """Fusionne la configuration du fichier avec celle par défaut"""
        for key, value in file_config.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _dict_to_obj(self, config_dict: Dict[str, Any]):
        """Convertit un dictionnaire en attributs d'objet"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, type('Config', (), {})())
                obj = getattr(self, key)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        setattr(obj, sub_key, type('Config', (), {})())
                        sub_obj = getattr(obj, sub_key)
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            setattr(sub_obj, sub_sub_key, sub_sub_value)
                    else:
                        setattr(obj, sub_key, sub_value)
            else:
                setattr(self, key, value)
