"""
Pydantic models for API requests and responses
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid

class FaceDetection(BaseModel):
    """Single face detection result"""
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence score")
    landmarks: Optional[List[float]] = Field(None, description="5-point facial landmarks")
    quality_score: Optional[float] = Field(None, description="Face quality assessment")
    liveness_score: Optional[float] = Field(None, description="Anti-spoofing score")

class PersonMetadata(BaseModel):
    """Person metadata for enrollment"""
    name: str = Field(..., description="Person's name")
    source: str = Field(..., description="Data source (upload, instagram, etc.)")
    consent_date: datetime = Field(default_factory=datetime.now)
    consent_type: str = Field("explicit", description="Type of consent given")
    tags: List[str] = Field(default_factory=list, description="Optional tags")
    notes: Optional[str] = Field(None, description="Additional notes")

class EnrollRequest(BaseModel):
    """Request to enroll a new person"""
    metadata: PersonMetadata
    photos: List[str] = Field(..., description="Base64 encoded images or file paths")
    
    @validator('photos')
    def validate_photos(cls, v):
        if len(v) == 0:
            raise ValueError("At least one photo is required")
        if len(v) > 50:
            raise ValueError("Maximum 50 photos allowed")
        return v

class EnrollResponse(BaseModel):
    """Response from enrollment"""
    person_id: str = Field(..., description="Unique person identifier")
    faces_processed: int = Field(..., description="Number of faces processed")
    faces_enrolled: int = Field(..., description="Number of faces successfully enrolled")
    embeddings_count: int = Field(..., description="Number of embeddings stored")
    quality_scores: List[float] = Field(..., description="Quality scores for each face")
    success: bool = Field(..., description="Whether enrollment was successful")
    message: str = Field(..., description="Status message")

class SearchMatch(BaseModel):
    """Single search result match"""
    person_id: str = Field(..., description="Matched person ID")
    name: str = Field(..., description="Person's name")
    similarity_score: float = Field(..., description="Cosine similarity score")
    source_image: Optional[str] = Field(None, description="Source image path/URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SearchRequest(BaseModel):
    """Request to search for faces"""
    image: str = Field(..., description="Base64 encoded image or file path")
    top_k: int = Field(10, ge=1, le=100, description="Number of top matches to return")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_metadata: bool = Field(True, description="Include person metadata in results")

class SearchResponse(BaseModel):
    """Response from face search"""
    faces_detected: int = Field(..., description="Number of faces detected in query image")
    matches: List[SearchMatch] = Field(..., description="Search results")
    processing_time: float = Field(..., description="Processing time in seconds")
    success: bool = Field(..., description="Whether search was successful")
    message: str = Field(..., description="Status message")

class VerifyRequest(BaseModel):
    """Request to verify if two images are the same person"""
    image1: str = Field(..., description="First image (base64 or path)")
    image2: str = Field(..., description="Second image (base64 or path)")
    threshold: Optional[float] = Field(None, description="Custom similarity threshold")

class VerifyResponse(BaseModel):
    """Response from face verification"""
    is_match: bool = Field(..., description="Whether faces match")
    similarity_score: float = Field(..., description="Similarity score")
    confidence: float = Field(..., description="Confidence in the result")
    faces_detected: Dict[str, int] = Field(..., description="Faces detected in each image")
    processing_time: float = Field(..., description="Processing time in seconds")
    success: bool = Field(..., description="Whether verification was successful")
    message: str = Field(..., description="Status message")

class PersonInfo(BaseModel):
    """Person information for listing"""
    person_id: str
    name: str
    enrollment_date: datetime
    faces_count: int
    source: str
    tags: List[str] = Field(default_factory=list)
    last_updated: datetime

class DeleteResponse(BaseModel):
    """Response from person deletion"""
    person_id: str = Field(..., description="Deleted person ID")
    embeddings_deleted: int = Field(..., description="Number of embeddings deleted")
    files_deleted: int = Field(..., description="Number of files deleted")
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")

class HealthResponse(BaseModel):
    """API health check response"""
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether face recognition model is loaded")
    database_connected: bool = Field(..., description="Database connection status")
    total_persons: int = Field(..., description="Total persons in database")
    total_embeddings: int = Field(..., description="Total face embeddings stored")
    uptime_seconds: float = Field(..., description="API uptime in seconds")

class StreamConfig(BaseModel):
    """Live stream processing configuration"""
    stream_url: str = Field(..., description="Stream URL or camera index")
    detection_frequency: int = Field(5, ge=1, le=30, description="Detect every N frames")
    tracking_enabled: bool = Field(True, description="Enable face tracking")
    alert_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Alert threshold for matches")
    save_matches: bool = Field(False, description="Save matched frames")

class ConsentRecord(BaseModel):
    """GDPR consent record"""
    person_id: str
    consent_date: datetime
    consent_type: str  # "explicit", "implied", "withdrawn"
    purpose: str  # "identification", "security", etc.
    retention_period: int  # days
    withdraw_date: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
