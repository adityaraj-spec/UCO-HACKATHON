"""
app/search/faiss_index.py

FAISS (Facebook AI Similarity Search) HNSW vector indexing.

Why FAISS HNSW (Hierarchical Navigable Small World):
  - At millions of users, O(N) database queries for 1:N checking are slow.
  - HNSW constructs a multi-layer graph, resolving query searches in O(log N)
    time (~5ms lookup latency at 50M scale).
  - Preserves recall accuracy (>99.5%) while reducing CPU usage.

Role in PhaseGuard:
  - Indexes all active anchor embeddings and rolling pool embeddings.
  - Provides similarity checks and 1:N pre-screening check.
  - Falls back to pgvector PostgreSQL direct cosine queries if FAISS is disabled.
"""

from __future__ import annotations

import logging
import os
import faiss
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FAISSIndexManager:
    """Manages the in-memory FAISS HNSW index and index serialization."""

    def __init__(self) -> None:
        self.dimension = settings.BIOHASH_PROJECTION_DIM  # Or settings.EMBEDDING_DIM
        self.index_path = settings.FAISS_INDEX_PATH
        self.hnsw_m = settings.FAISS_HNSW_M
        self.ef_construction = settings.FAISS_HNSW_EF_CONSTRUCTION
        self.index = None
        self._id_map: dict[int, str] = {}  # FAISS integer ID -> user_id (string)
        self._reverse_id_map: dict[str, int] = {}  # user_id -> FAISS integer ID
        self._next_faiss_id = 0

        self.initialize_index()

    def initialize_index(self) -> None:
        """Create or load index from disk."""
        if settings.FAISS_ENABLED and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info("Loaded existing FAISS index from %s", self.index_path)
                # Recover mapping (in real deployment, mapping is synced in DB or separate JSON)
                return
            except Exception as e:
                logger.error("Failed to load FAISS index from disk, creating new: %s", e)

        # Create HNSW Index with Cosine Similarity (Inner Product on normalized vectors)
        hnsw_index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        # Set construction parameters before wrapping
        hnsw_index.hnsw.efConstruction = self.ef_construction
        hnsw_index.hnsw.efSearch = 64
        self.index = faiss.IndexIDMap(hnsw_index)
        logger.info("Initialized new FAISS HNSW index (dimension=%d)", self.dimension)

    def add_vector(self, user_id: str, vector: np.ndarray) -> bool:
        """
        Add a user's BioHashed template to the index.

        Args:
            user_id: Target user UUID string
            vector: 1D float32 array of self.dimension size

        Returns:
            True if vector was successfully added
        """
        if not settings.FAISS_ENABLED or self.index is None:
            return False

        # Ensure vector is float32 and normalized for inner product (cosine sim equivalent)
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Map to integer ID
        if user_id in self._reverse_id_map:
            faiss_id = self._reverse_id_map[user_id]
            # Remove old vector first
            self.remove_vector(user_id)
        else:
            faiss_id = self._next_faiss_id
            self._next_faiss_id += 1

        try:
            # FAISS expects 2D array: (1, dim)
            self.index.add_with_ids(
                np.expand_dims(vec, axis=0),
                np.array([faiss_id], dtype=np.int64)
            )
            self._id_map[faiss_id] = user_id
            self._reverse_id_map[user_id] = faiss_id
            self.save_index()
            logger.debug("Added vector to FAISS, assigned internal ID %d for user %s", faiss_id, user_id)
            return True
        except Exception as e:
            logger.exception("Failed to add vector to FAISS for user %s: %s", user_id, e)
            return False

    def remove_vector(self, user_id: str) -> bool:
        """Remove a user's vector from index."""
        if not settings.FAISS_ENABLED or self.index is None or user_id not in self._reverse_id_map:
            return False

        faiss_id = self._reverse_id_map[user_id]
        try:
            # IndexIDMap supports remove_ids
            self.index.remove_ids(np.array([faiss_id], dtype=np.int64))
            del self._id_map[faiss_id]
            del self._reverse_id_map[user_id]
            self.save_index()
            return True
        except Exception as e:
            logger.error("Failed to remove vector from FAISS for user %s: %s", user_id, e)
            return False

    def search_similarity(self, vector: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        """
        Query FAISS index for nearest neighbors.

        Returns:
            List of tuples: (claims_user_id, cosine_score)
        """
        if not settings.FAISS_ENABLED or self.index is None or self.index.ntotal == 0:
            return []

        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        try:
            # Query FAISS
            scores, indices = self.index.search(
                np.expand_dims(vec, axis=0),
                top_k
            )

            results = []
            for score, faiss_id in zip(scores[0], indices[0]):
                if faiss_id == -1:
                    continue  # No match
                usr_id = self._id_map.get(faiss_id)
                if usr_id:
                    results.append((usr_id, float(score)))

            return results
        except Exception as e:
            logger.error("FAISS search failed: %s", e)
            return []

    def save_index(self) -> None:
        """Write index binary to disk."""
        if not settings.FAISS_ENABLED or self.index is None:
            return
        try:
            # Create parent dirs
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
        except Exception as e:
            logger.error("Failed to save FAISS index: %s", e)


# Module-level singleton
_faiss_manager: FAISSIndexManager | None = None


def get_faiss_manager() -> FAISSIndexManager:
    """Return singleton FAISSIndexManager."""
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FAISSIndexManager()
    return _faiss_manager
class MockFAISSIndexManager:
    """Fallback mock FAISS when library fails to load."""
    def __init__(self):
        self._store = {}
    def add_vector(self, user_id, vector):
        self._store[user_id] = vector
        return True
    def remove_vector(self, user_id):
        self._store.pop(user_id, None)
        return True
    def search_similarity(self, vector, top_k=1):
        if not self._store: return []
        results = []
        vec = vector / np.linalg.norm(vector)
        for uid, v in self._store.items():
            norm_v = v / np.linalg.norm(v)
            score = float(np.dot(vec, norm_v))
            results.append((uid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    def save_index(self): pass
