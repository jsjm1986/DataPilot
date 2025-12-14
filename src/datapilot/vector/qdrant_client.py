# -*- coding: utf-8 -*-
"""
Qdrant Vector Store Client
Supports both local file mode and server mode
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchParams, HnswConfigDiff
)

from ..config.settings import get_settings


# Global client instance (shared across all collections)
_shared_client: Optional[QdrantClient] = None
_shared_client_path: Optional[str] = None


def get_shared_client(path: str = 'data/qdrant') -> QdrantClient:
    """Get shared Qdrant client instance for local mode"""
    global _shared_client, _shared_client_path
    if _shared_client is None or _shared_client_path != path:
        # Use in-memory mode on Windows to avoid path issues
        import platform
        if platform.system() == 'Windows':
            # Use in-memory mode for Windows
            _shared_client = QdrantClient(":memory:")
            _shared_client_path = ":memory:"
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            _shared_client = QdrantClient(path=path)
            _shared_client_path = path
    return _shared_client


class QdrantStore:
    """
    Qdrant Vector Store Wrapper
    
    Supports:
    - Local file mode (for development)
    - Server mode (for production)
    """
    
    def __init__(
        self,
        collection_name: str = 'datapilot',
        vector_size: int = 1024,  # bge-large-zh-v1.5 dimension
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Determine mode: local file (default) or server (explicit url)
        # Use local mode unless explicit url is provided
        if url:
            # Server mode - only when explicit url is provided
            self._client = QdrantClient(
                url=url,
                api_key=settings.qdrant_api_key or None,
            )
            self._mode = 'server'
        else:
            # Local file mode (default for development)
            # Use shared client to avoid lock conflicts
            # On Windows, use in-memory mode to avoid path issues
            import platform
            if platform.system() == 'Windows':
                self._client = get_shared_client(":memory:")
                self._mode = 'memory'
            else:
                local_path = path or 'data/qdrant'
                Path(local_path).mkdir(parents=True, exist_ok=True)
                self._client = get_shared_client(local_path)
                self._mode = 'local'
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if not exists"""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            settings = get_settings()
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=settings.qdrant_hnsw_m,
                    ef_construct=128,
                ),
            )
    
    @property
    def client(self) -> QdrantClient:
        """Get raw Qdrant client"""
        return self._client
    
    @property
    def mode(self) -> str:
        """Get current mode (local/server)"""
        return self._mode
    
    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Upsert vectors with payloads
        
        Args:
            ids: Point IDs
            vectors: Vector embeddings
            payloads: Optional metadata for each point
        """
        points = []
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            payload = payloads[i] if payloads else {}
            points.append(PointStruct(
                id=hash(id_) % (2**63),  # Convert string ID to int
                vector=vec,
                payload={'_id': id_, **payload},
            ))
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search similar vectors
        
        Args:
            query_vector: Query embedding
            limit: Max results
            score_threshold: Min similarity score
            filter_conditions: Optional filter
            
        Returns:
            List of results with id, score, payload
        """
        # Build filter
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                ))
            query_filter = Filter(must=conditions)
        
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )
        
        return [
            {
                'id': r.payload.get('_id', str(r.id)),
                'score': r.score,
                'payload': {k: v for k, v in r.payload.items() if k != '_id'},
            }
            for r in results
        ]
    
    def delete(self, ids: List[str]) -> None:
        """Delete points by IDs"""
        int_ids = [hash(id_) % (2**63) for id_ in ids]
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=int_ids,
        )
    
    def count(self) -> int:
        """Get total point count"""
        info = self._client.get_collection(self.collection_name)
        return info.points_count
    
    def clear(self) -> None:
        """Clear all points in collection"""
        self._client.delete_collection(self.collection_name)
        self._ensure_collection()


# Global store instance
_store: Optional[QdrantStore] = None


def get_qdrant_store() -> QdrantStore:
    """Get global Qdrant store instance"""
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


__all__ = ['QdrantStore', 'get_qdrant_store']
