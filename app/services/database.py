"""
Vector database service for face embeddings storage and search
Supports both FAISS (local) and Qdrant (remote) backends
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import time
from datetime import datetime, timedelta
import uuid
import pickle

try:
    import faiss
except ImportError:
    faiss = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Base class for vector database implementations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_config = config.get('database', {})
        
    def add_person(self, person_id: str, embeddings: List[np.ndarray], 
                   metadata: Dict) -> bool:
        """Add a person with their face embeddings"""
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               threshold: float = 0.5) -> List[Dict]:
        """Search for similar faces"""
        raise NotImplementedError
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a person and all their embeddings"""
        raise NotImplementedError
    
    def get_person_info(self, person_id: str) -> Optional[Dict]:
        """Get person information"""
        raise NotImplementedError
    
    def list_persons(self) -> List[Dict]:
        """List all persons in database"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        raise NotImplementedError

class FAISSDatabase(VectorDatabase):
    """FAISS-based local vector database"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if faiss is None:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        faiss_config = self.db_config.get('faiss', {})
        self.index_path = Path(faiss_config.get('index_path', 'storage/database/face_index.faiss'))
        self.metadata_path = Path(faiss_config.get('metadata_path', 'storage/database/metadata.json'))
        
        # Create directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = {}
        self.embedding_to_person = {}  # Maps embedding index to person_id
        self.person_embeddings = {}    # Maps person_id to list of embedding indices
        
        self._load_or_create_index()
        logger.info(f"FAISS database initialized with {self.index.ntotal} embeddings")
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get('persons', {})
                    self.embedding_to_person = data.get('embedding_to_person', {})
                    self.person_embeddings = data.get('person_embeddings', {})
                
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} embeddings")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
        
        # Create new index (assuming 512-dimensional embeddings)
        self.index = faiss.IndexFlatIP(512)  # Inner product for cosine similarity
        self.metadata = {}
        self.embedding_to_person = {}
        self.person_embeddings = {}
        
        logger.info("Created new FAISS index")
    
    def add_person(self, person_id: str, embeddings: List[np.ndarray], 
                   metadata: Dict) -> bool:
        """Add a person with their face embeddings"""
        try:
            if not embeddings:
                return False
            
            # Normalize embeddings for cosine similarity
            embeddings_array = np.vstack([emb / np.linalg.norm(emb) for emb in embeddings])
            embeddings_array = embeddings_array.astype(np.float32)
            
            # Get starting index for new embeddings
            start_idx = self.index.ntotal
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Update metadata
            self.metadata[person_id] = {
                'name': metadata.get('name', ''),
                'source': metadata.get('source', ''),
                'enrollment_date': metadata.get('enrollment_date', datetime.now().isoformat()),
                'consent_date': metadata.get('consent_date', datetime.now().isoformat()),
                'consent_type': metadata.get('consent_type', 'explicit'),
                'tags': metadata.get('tags', []),
                'notes': metadata.get('notes', ''),
                'faces_count': len(embeddings),
                'last_updated': datetime.now().isoformat()
            }
            
            # Update embedding mappings
            embedding_indices = list(range(start_idx, start_idx + len(embeddings)))
            self.person_embeddings[person_id] = embedding_indices
            
            for idx in embedding_indices:
                self.embedding_to_person[str(idx)] = person_id
            
            self._save_index()
            logger.info(f"Added person {person_id} with {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add person {person_id}: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               threshold: float = 0.5) -> List[Dict]:
        """Search for similar faces"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_emb = query_embedding / np.linalg.norm(query_embedding)
            query_emb = query_emb.astype(np.float32).reshape(1, -1)
            
            # Search in FAISS
            scores, indices = self.index.search(query_emb, min(top_k * 5, self.index.ntotal))
            
            # Group results by person and aggregate scores
            person_scores = {}
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < threshold:
                    continue
                
                person_id = self.embedding_to_person.get(str(idx))
                if person_id:
                    if person_id not in person_scores:
                        person_scores[person_id] = []
                    person_scores[person_id].append(float(score))
            
            # Create results with aggregated scores
            results = []
            for person_id, scores_list in person_scores.items():
                if person_id in self.metadata:
                    person_data = self.metadata[person_id]
                    
                    # Use maximum score as the person's similarity
                    max_score = max(scores_list)
                    avg_score = sum(scores_list) / len(scores_list)
                    
                    results.append({
                        'person_id': person_id,
                        'name': person_data.get('name', ''),
                        'similarity_score': max_score,
                        'avg_similarity': avg_score,
                        'match_count': len(scores_list),
                        'source': person_data.get('source', ''),
                        'tags': person_data.get('tags', []),
                        'enrollment_date': person_data.get('enrollment_date', ''),
                        'metadata': person_data
                    })
            
            # Sort by similarity score and return top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a person and all their embeddings"""
        try:
            if person_id not in self.metadata:
                return False
            
            # Get embedding indices for this person
            embedding_indices = self.person_embeddings.get(person_id, [])
            
            # Remove from metadata
            del self.metadata[person_id]
            del self.person_embeddings[person_id]
            
            # Remove from embedding mapping
            for idx in embedding_indices:
                if str(idx) in self.embedding_to_person:
                    del self.embedding_to_person[str(idx)]
            
            # Note: FAISS doesn't support deletion of individual vectors efficiently
            # In production, you might want to rebuild the index periodically
            # For now, we just mark them as deleted in metadata
            
            self._save_index()
            logger.info(f"Deleted person {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete person {person_id}: {e}")
            return False
    
    def get_person_info(self, person_id: str) -> Optional[Dict]:
        """Get person information"""
        return self.metadata.get(person_id)
    
    def list_persons(self) -> List[Dict]:
        """List all persons in database"""
        persons = []
        for person_id, data in self.metadata.items():
            person_info = {
                'person_id': person_id,
                'name': data.get('name', ''),
                'enrollment_date': data.get('enrollment_date', ''),
                'faces_count': data.get('faces_count', 0),
                'source': data.get('source', ''),
                'tags': data.get('tags', []),
                'last_updated': data.get('last_updated', '')
            }
            persons.append(person_info)
        
        return sorted(persons, key=lambda x: x['enrollment_date'], reverse=True)
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_persons': len(self.metadata),
            'total_embeddings': self.index.ntotal if self.index else 0,
            'index_size_mb': self.index_path.stat().st_size / (1024*1024) if self.index_path.exists() else 0,
            'last_updated': max([data.get('last_updated', '') for data in self.metadata.values()]) if self.metadata else ''
        }
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            data = {
                'persons': self.metadata,
                'embedding_to_person': self.embedding_to_person,
                'person_embeddings': self.person_embeddings,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

class QdrantDatabase(VectorDatabase):
    """Qdrant-based remote vector database"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if QdrantClient is None:
            raise ImportError("Qdrant client not installed. Run: pip install qdrant-client")
        
        qdrant_config = self.db_config.get('qdrant', {})
        self.client = QdrantClient(url=qdrant_config.get('url', 'http://localhost:6333'))
        self.collection_name = qdrant_config.get('collection_name', 'faces')
        self.vector_size = qdrant_config.get('vector_size', 512)
        
        self._ensure_collection()
        logger.info(f"Qdrant database initialized: {self.collection_name}")
    
    def _ensure_collection(self):
        """Ensure collection exists"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def add_person(self, person_id: str, embeddings: List[np.ndarray], 
                   metadata: Dict) -> bool:
        """Add a person with their face embeddings"""
        try:
            points = []
            for i, embedding in enumerate(embeddings):
                point_id = f"{person_id}_{i}_{int(time.time())}"
                
                point_metadata = {
                    'person_id': person_id,
                    'embedding_index': i,
                    **metadata,
                    'enrollment_date': datetime.now().isoformat()
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=point_metadata
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added person {person_id} with {len(embeddings)} embeddings to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add person {person_id} to Qdrant: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               threshold: float = 0.5) -> List[Dict]:
        """Search for similar faces"""
        try:
            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k * 5,  # Get more results for grouping
                score_threshold=threshold
            )
            
            # Group by person_id and aggregate scores
            person_scores = {}
            for hit in search_result:
                person_id = hit.payload.get('person_id')
                score = hit.score
                
                if person_id not in person_scores:
                    person_scores[person_id] = {
                        'scores': [],
                        'metadata': hit.payload
                    }
                person_scores[person_id]['scores'].append(score)
            
            # Create aggregated results
            results = []
            for person_id, data in person_scores.items():
                scores = data['scores']
                metadata = data['metadata']
                
                results.append({
                    'person_id': person_id,
                    'name': metadata.get('name', ''),
                    'similarity_score': max(scores),
                    'avg_similarity': sum(scores) / len(scores),
                    'match_count': len(scores),
                    'source': metadata.get('source', ''),
                    'tags': metadata.get('tags', []),
                    'enrollment_date': metadata.get('enrollment_date', ''),
                    'metadata': metadata
                })
            
            # Sort and return top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a person and all their embeddings"""
        try:
            # Delete all points with this person_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "person_id", "match": {"value": person_id}}
                        ]
                    }
                }
            )
            
            logger.info(f"Deleted person {person_id} from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete person {person_id} from Qdrant: {e}")
            return False
    
    def get_person_info(self, person_id: str) -> Optional[Dict]:
        """Get person information"""
        try:
            # Search for any point with this person_id
            result = self.client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * self.vector_size,  # Dummy vector
                limit=1,
                query_filter={
                    "must": [
                        {"key": "person_id", "match": {"value": person_id}}
                    ]
                }
            )
            
            if result:
                return result[0].payload
            return None
            
        except Exception as e:
            logger.error(f"Failed to get person info for {person_id}: {e}")
            return None
    
    def list_persons(self) -> List[Dict]:
        """List all persons in database"""
        # This is simplified - in production you'd want proper aggregation
        try:
            # Get all unique person_ids (this is a simplified approach)
            # In production, you'd maintain a separate persons collection
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Adjust based on your needs
                with_payload=True,
                with_vectors=False
            )
            
            persons = {}
            for point in result[0]:
                person_id = point.payload.get('person_id')
                if person_id and person_id not in persons:
                    persons[person_id] = {
                        'person_id': person_id,
                        'name': point.payload.get('name', ''),
                        'enrollment_date': point.payload.get('enrollment_date', ''),
                        'source': point.payload.get('source', ''),
                        'tags': point.payload.get('tags', []),
                        'faces_count': 1
                    }
                elif person_id:
                    persons[person_id]['faces_count'] += 1
            
            return list(persons.values())
            
        except Exception as e:
            logger.error(f"Failed to list persons: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'total_embeddings': info.points_count,
                'total_persons': len(self.list_persons()),  # This is expensive
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count
            }
        except Exception as e:
            logger.error(f"Failed to get Qdrant stats: {e}")
            return {}

def create_database(config: Dict) -> VectorDatabase:
    """Factory function to create appropriate database backend"""
    db_type = config.get('database', {}).get('type', 'faiss').lower()
    
    if db_type == 'faiss':
        return FAISSDatabase(config)
    elif db_type == 'qdrant':
        return QdrantDatabase(config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
