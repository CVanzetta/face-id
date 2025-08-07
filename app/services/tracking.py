"""
Enhanced video tracking service with ByteTrack-inspired algorithm
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)

class TrackState:
    """Track state enumeration"""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3

class FaceTrack:
    """Enhanced face track with temporal smoothing and state management"""
    
    def __init__(self, track_id: int, detection: Dict, smooth_window: int = 7):
        self.id = track_id
        self.state = TrackState.NEW
        self.smooth_window = smooth_window
        
        # Detection history
        self.detections = deque([detection], maxlen=smooth_window)
        self.embeddings = deque([detection['embedding']], maxlen=smooth_window)
        self.confidences = deque([detection['confidence']], maxlen=smooth_window)
        self.quality_scores = deque([detection.get('quality_score', 0.5)], maxlen=smooth_window)
        
        # Tracking state
        self.bbox = detection['bbox']
        self.last_detection_frame = 0
        self.missed_frames = 0
        self.total_frames = 1
        
        # Identity information
        self.person_id = None
        self.person_name = "Unknown"
        self.similarity_score = 0.0
        self.identification_confidence = 0.0
        
        # Timestamps
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Kalman filter for smooth tracking (simplified)
        self.velocity = [0, 0]  # [dx, dy] per frame
        
    def update(self, detection: Dict, frame_id: int):
        """Update track with new detection"""
        self.detections.append(detection)
        self.embeddings.append(detection['embedding'])
        self.confidences.append(detection['confidence'])
        self.quality_scores.append(detection.get('quality_score', 0.5))
        
        # Update position and velocity
        old_center = self._get_bbox_center(self.bbox)
        new_bbox = detection['bbox']
        new_center = self._get_bbox_center(new_bbox)
        
        self.velocity = [
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        ]
        
        self.bbox = new_bbox
        self.last_detection_frame = frame_id
        self.last_update_time = time.time()
        self.missed_frames = 0
        self.total_frames += 1
        
        # Update state
        if self.state == TrackState.NEW:
            self.state = TrackState.TRACKED
    
    def predict_next_bbox(self) -> List[float]:
        """Predict next bounding box position based on velocity"""
        if self.missed_frames == 0:
            return self.bbox
        
        # Simple linear prediction
        center = self._get_bbox_center(self.bbox)
        predicted_center = [
            center[0] + self.velocity[0] * self.missed_frames,
            center[1] + self.velocity[1] * self.missed_frames
        ]
        
        # Maintain bbox size
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        
        return [
            predicted_center[0] - w/2,
            predicted_center[1] - h/2,
            predicted_center[0] + w/2,
            predicted_center[1] + h/2
        ]
    
    def mark_missed(self, frame_id: int):
        """Mark track as missed for this frame"""
        self.missed_frames += 1
        
        # Update predicted position
        self.bbox = self.predict_next_bbox()
        
        # Update state based on missed frames
        if self.missed_frames > 30:  # Lost after 30 frames
            self.state = TrackState.LOST
    
    def get_averaged_embedding(self) -> np.ndarray:
        """Get temporally averaged embedding"""
        if not self.embeddings:
            return np.zeros(512)
        
        # Weight recent embeddings more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.embeddings)))
        weights = weights / np.sum(weights)
        
        embeddings_array = np.array(self.embeddings)
        weighted_embedding = np.average(embeddings_array, axis=0, weights=weights)
        
        # Normalize
        return weighted_embedding / np.linalg.norm(weighted_embedding)
    
    def get_average_confidence(self) -> float:
        """Get average confidence score"""
        return float(np.mean(self.confidences))
    
    def get_average_quality(self) -> float:
        """Get average quality score"""
        return float(np.mean(self.quality_scores))
    
    def update_identity(self, person_id: str, person_name: str, 
                       similarity_score: float, confidence: float):
        """Update track identity information"""
        self.person_id = person_id
        self.person_name = person_name
        self.similarity_score = similarity_score
        self.identification_confidence = confidence
    
    def get_track_info(self) -> Dict:
        """Get comprehensive track information"""
        return {
            'track_id': self.id,
            'state': self.state,
            'bbox': self.bbox,
            'predicted_bbox': self.predict_next_bbox(),
            'person_id': self.person_id,
            'person_name': self.person_name,
            'similarity_score': self.similarity_score,
            'identification_confidence': self.identification_confidence,
            'average_confidence': self.get_average_confidence(),
            'average_quality': self.get_average_quality(),
            'total_frames': self.total_frames,
            'missed_frames': self.missed_frames,
            'track_duration': time.time() - self.start_time,
            'last_update': self.last_update_time
        }
    
    @staticmethod
    def _get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

class VideoFaceTracker:
    """Enhanced multi-object tracker for faces"""
    
    def __init__(self, config: Dict):
        self.config = config
        video_config = config.get('video', {})
        
        # Tracking parameters
        self.iou_threshold = 0.3
        self.max_age = video_config.get('track_max_age', 15)
        self.smooth_window = video_config.get('smooth_window', 7)
        self.min_hits = 3  # Minimum hits before confirming track
        
        # State
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        
        logger.info("Video face tracker initialized")
    
    def update(self, detections: List[Dict]) -> List[FaceTrack]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of face detections with bbox, confidence, embedding, etc.
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict track positions
        for track in self.tracks:
            track.predict_next_bbox()
        
        # Associate detections with tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate_detections(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx].update(detections[det_idx], self.frame_count)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed(self.frame_count)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = FaceTrack(
                track_id=self.next_track_id,
                detection=detections[det_idx],
                smooth_window=self.smooth_window
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_age]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.total_frames >= self.min_hits or t.state == TrackState.TRACKED]
    
    def _associate_detections(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with existing tracks using IoU and embedding similarity
        
        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            predicted_bbox = track.predict_next_bbox()
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._compute_iou(predicted_bbox, detection['bbox'])
        
        # Compute embedding similarity matrix
        embedding_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            track_embedding = track.get_averaged_embedding()
            for d, detection in enumerate(detections):
                det_embedding = detection['embedding']
                if track_embedding is not None and det_embedding is not None:
                    # Normalize embeddings
                    track_emb_norm = track_embedding / np.linalg.norm(track_embedding)
                    det_emb_norm = det_embedding / np.linalg.norm(det_embedding)
                    similarity = np.dot(track_emb_norm, det_emb_norm)
                    embedding_matrix[t, d] = max(0, similarity)  # Only positive similarities
        
        # Combined cost matrix (IoU + embedding similarity)
        cost_matrix = 0.6 * iou_matrix + 0.4 * embedding_matrix
        
        # Hungarian assignment (simplified greedy approach)
        matched_pairs = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Greedy assignment
        while len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            # Find best match
            best_score = 0
            best_track = -1
            best_detection = -1
            
            for t in unmatched_tracks:
                for d in unmatched_detections:
                    if (cost_matrix[t, d] > best_score and 
                        iou_matrix[t, d] > self.iou_threshold):
                        best_score = cost_matrix[t, d]
                        best_track = t
                        best_detection = d
            
            if best_track >= 0:
                matched_pairs.append((best_track, best_detection))
                unmatched_tracks.remove(best_track)
                unmatched_detections.remove(best_detection)
            else:
                break
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    @staticmethod
    def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union of two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def get_active_tracks(self) -> List[FaceTrack]:
        """Get all active tracks"""
        return [t for t in self.tracks if t.state in [TrackState.NEW, TrackState.TRACKED]]
    
    def get_track_by_id(self, track_id: int) -> Optional[FaceTrack]:
        """Get track by ID"""
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        logger.info("Video tracker reset")
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        active_tracks = self.get_active_tracks()
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len(active_tracks),
            'frame_count': self.frame_count,
            'next_track_id': self.next_track_id,
            'tracks_by_state': {
                'new': len([t for t in self.tracks if t.state == TrackState.NEW]),
                'tracked': len([t for t in self.tracks if t.state == TrackState.TRACKED]),
                'lost': len([t for t in self.tracks if t.state == TrackState.LOST]),
            }
        }
