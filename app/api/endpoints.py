"""
FastAPI endpoints for Face Recognition API
"""
import os
import uuid
import base64
import io
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from ..models.schemas import (
    EnrollRequest, EnrollResponse, SearchRequest, SearchResponse,
    VerifyRequest, VerifyResponse, DeleteResponse, HealthResponse,
    PersonInfo, StreamConfig
)
from ..core.face_engine import FaceRecognitionEngine
from ..services.database import create_database, VectorDatabase
from ..services.tracking import VideoFaceTracker
from .dependencies import get_face_engine, get_database, get_config

logger = logging.getLogger(__name__)

# Global variables for model loading
face_engine: Optional[FaceRecognitionEngine] = None
database: Optional[VectorDatabase] = None
config: Optional[Dict] = None
startup_time: float = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global face_engine, database, config, startup_time
    
    startup_time = time.time()
    
    try:
        # Load configuration
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Initialize face engine
        face_engine = FaceRecognitionEngine(config)
        logger.info("Face recognition engine loaded")
        
        # Initialize database
        database = create_database(config)
        logger.info("Database initialized")
        
        # Create storage directories
        storage_config = config.get('storage', {})
        uploads_dir = Path(storage_config.get('uploads_dir', 'storage/uploads'))
        temp_dir = Path(storage_config.get('temp_dir', 'storage/temp'))
        
        uploads_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Face Recognition API startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Face Recognition API shutting down")

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="GDPR-compliant face recognition system with InsightFace and FAISS/Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Face Recognition API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check(
    db: VectorDatabase = Depends(get_database),
    engine: FaceRecognitionEngine = Depends(get_face_engine)
):
    """Health check endpoint"""
    try:
        db_stats = db.get_stats()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model_loaded=engine is not None,
            database_connected=True,
            total_persons=db_stats.get('total_persons', 0),
            total_embeddings=db_stats.get('total_embeddings', 0),
            uptime_seconds=time.time() - startup_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            model_loaded=engine is not None,
            database_connected=False,
            total_persons=0,
            total_embeddings=0,
            uptime_seconds=time.time() - startup_time
        )

@app.post("/enroll", response_model=EnrollResponse)
async def enroll_person(
    name: str = Form(...),
    source: str = Form("upload"),
    consent_type: str = Form("explicit"),
    tags: str = Form(""),
    notes: str = Form(""),
    files: List[UploadFile] = File(...),
    engine: FaceRecognitionEngine = Depends(get_face_engine),
    db: VectorDatabase = Depends(get_database),
    cfg: Dict = Depends(get_config)
):
    """
    Enroll a new person with face images
    
    Args:
        name: Person's name
        source: Data source (upload, instagram, etc.)
        consent_type: Type of consent (explicit, implied)
        tags: Comma-separated tags
        notes: Additional notes
        files: Image files to process
    """
    try:
        person_id = str(uuid.uuid4())
        embeddings = []
        quality_scores = []
        faces_processed = 0
        
        # Process uploaded files
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
                
            # Read image
            content = await file.read()
            image = Image.open(io.BytesIO(content))
            image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = engine.detect_faces(image_array)
            faces_processed += len(faces)
            
            for face in faces:
                # Quality check
                quality_threshold = cfg.get('thresholds', {}).get('face_quality', 0.5)
                if face.get('quality_score', 0) < quality_threshold:
                    continue
                
                # Liveness check
                liveness_threshold = cfg.get('thresholds', {}).get('liveness_score', 0.7)
                if face.get('liveness_score', 0) < liveness_threshold:
                    continue
                
                embeddings.append(face['embedding'])
                quality_scores.append(face.get('quality_score', 0.5))
        
        if not embeddings:
            return EnrollResponse(
                person_id=person_id,
                faces_processed=faces_processed,
                faces_enrolled=0,
                embeddings_count=0,
                quality_scores=[],
                success=False,
                message="No valid faces found in uploaded images"
            )
        
        # Prepare metadata
        metadata = {
            'name': name,
            'source': source,
            'consent_type': consent_type,
            'tags': [tag.strip() for tag in tags.split(',') if tag.strip()],
            'notes': notes
        }
        
        # Store in database
        success = db.add_person(person_id, embeddings, metadata)
        
        if success:
            logger.info(f"Successfully enrolled person {name} with {len(embeddings)} embeddings")
            return EnrollResponse(
                person_id=person_id,
                faces_processed=faces_processed,
                faces_enrolled=len(embeddings),
                embeddings_count=len(embeddings),
                quality_scores=quality_scores,
                success=True,
                message=f"Successfully enrolled {name}"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to store person in database")
            
    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_faces(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    threshold: Optional[float] = Form(None),
    engine: FaceRecognitionEngine = Depends(get_face_engine),
    db: VectorDatabase = Depends(get_database),
    cfg: Dict = Depends(get_config)
):
    """
    Search for faces in the database
    
    Args:
        file: Image file to search
        top_k: Number of top matches to return
        threshold: Minimum similarity threshold
    """
    try:
        start_time = time.time()
        
        # Default threshold
        if threshold is None:
            threshold = cfg.get('thresholds', {}).get('cosine_similarity', 0.35)
        
        # Read and process image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = engine.detect_faces(image_array)
        
        if not faces:
            return SearchResponse(
                faces_detected=0,
                matches=[],
                processing_time=time.time() - start_time,
                success=True,
                message="No faces detected in image"
            )
        
        # Use the best quality face for search
        best_face = max(faces, key=lambda x: x.get('quality_score', 0))
        query_embedding = best_face['embedding']
        
        # Search in database
        results = db.search(query_embedding, top_k, threshold)
        
        # Format results
        matches = []
        for result in results:
            matches.append({
                'person_id': result['person_id'],
                'name': result['name'],
                'similarity_score': result['similarity_score'],
                'source_image': None,  # Could add source image path
                'metadata': {
                    'source': result.get('source', ''),
                    'tags': result.get('tags', []),
                    'enrollment_date': result.get('enrollment_date', ''),
                    'match_count': result.get('match_count', 1)
                }
            })
        
        return SearchResponse(
            faces_detected=len(faces),
            matches=matches,
            processing_time=time.time() - start_time,
            success=True,
            message=f"Found {len(matches)} matches"
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/verify", response_model=VerifyResponse)
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    engine: FaceRecognitionEngine = Depends(get_face_engine),
    cfg: Dict = Depends(get_config)
):
    """
    Verify if two images contain the same person
    
    Args:
        file1: First image file
        file2: Second image file
        threshold: Custom similarity threshold
    """
    try:
        start_time = time.time()
        
        # Default threshold
        if threshold is None:
            threshold = cfg.get('thresholds', {}).get('cosine_similarity', 0.35)
        
        # Read images
        content1 = await file1.read()
        content2 = await file2.read()
        
        image1 = Image.open(io.BytesIO(content1))
        image2 = Image.open(io.BytesIO(content2))
        
        image1_array = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image2_array = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        
        # Verify faces
        result = engine.verify_faces(image1_array, image2_array, threshold)
        
        return VerifyResponse(
            is_match=result['is_match'],
            similarity_score=result['similarity_score'],
            confidence=result['confidence'],
            faces_detected={
                'image1': result['faces_detected']['image1'],
                'image2': result['faces_detected']['image2']
            },
            processing_time=time.time() - start_time,
            success=True,
            message="Verification completed"
        )
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.delete("/person/{person_id}", response_model=DeleteResponse)
async def delete_person(
    person_id: str,
    db: VectorDatabase = Depends(get_database)
):
    """
    Delete a person and all their data (GDPR right to be forgotten)
    
    Args:
        person_id: Person ID to delete
    """
    try:
        # Get person info before deletion
        person_info = db.get_person_info(person_id)
        if not person_info:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Delete from database
        success = db.delete_person(person_id)
        
        if success:
            # TODO: Delete associated files from storage
            files_deleted = 0  # Implement file cleanup
            
            logger.info(f"Deleted person {person_id}")
            return DeleteResponse(
                person_id=person_id,
                embeddings_deleted=person_info.get('faces_count', 0),
                files_deleted=files_deleted,
                success=True,
                message=f"Successfully deleted person {person_id}"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to delete person")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.get("/persons", response_model=List[PersonInfo])
async def list_persons(
    db: VectorDatabase = Depends(get_database)
):
    """List all persons in the database"""
    try:
        persons = db.list_persons()
        
        # Convert to PersonInfo format
        result = []
        for person in persons:
            result.append(PersonInfo(
                person_id=person['person_id'],
                name=person['name'],
                enrollment_date=person['enrollment_date'],
                faces_count=person['faces_count'],
                source=person['source'],
                tags=person.get('tags', []),
                last_updated=person.get('last_updated', person['enrollment_date'])
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to list persons: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list persons: {str(e)}")

@app.get("/person/{person_id}")
async def get_person(
    person_id: str,
    db: VectorDatabase = Depends(get_database)
):
    """Get detailed information about a person"""
    try:
        person_info = db.get_person_info(person_id)
        
        if not person_info:
            raise HTTPException(status_code=404, detail="Person not found")
        
        return person_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get person {person_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get person: {str(e)}")

@app.get("/stats")
async def get_stats(
    db: VectorDatabase = Depends(get_database)
):
    """Get system statistics"""
    try:
        db_stats = db.get_stats()
        
        return {
            'database': db_stats,
            'uptime_seconds': time.time() - startup_time,
            'version': '1.0.0'
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )
