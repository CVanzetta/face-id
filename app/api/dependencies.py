"""
FastAPI dependencies for dependency injection
"""
from typing import Dict
import yaml
from fastapi import HTTPException

from ..core.face_engine import FaceRecognitionEngine
from ..services.database import VectorDatabase

# Global instances (initialized in lifespan)
_face_engine: FaceRecognitionEngine = None
_database: VectorDatabase = None
_config: Dict = None

def set_global_instances(engine: FaceRecognitionEngine, db: VectorDatabase, config: Dict):
    """Set global instances (called during app startup)"""
    global _face_engine, _database, _config
    _face_engine = engine
    _database = db
    _config = config

def get_face_engine() -> FaceRecognitionEngine:
    """Get face recognition engine instance"""
    if _face_engine is None:
        raise HTTPException(status_code=503, detail="Face recognition engine not available")
    return _face_engine

def get_database() -> VectorDatabase:
    """Get database instance"""
    if _database is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return _database

def get_config() -> Dict:
    """Get configuration"""
    if _config is None:
        # Fallback to loading config
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            raise HTTPException(status_code=503, detail="Configuration not available")
    return _config
