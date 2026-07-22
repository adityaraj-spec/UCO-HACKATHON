"""
app/services/faiss_service.py

High-performance FAISS index manager for scalable million-user voiceprint vector search.
Uses IndexIVFPQ + IndexHNSWFlat coarse quantizer, double-buffer index updates, and Redis mapping.
"""

import os
import json
import logging
import uuid
import threading
from typing import Dict, List, Tuple, Optional
import numpy as np
import faiss
from sqlalchemy.future import select

from app.core.config import get_settings
from app.models.voiceprint import Voiceprint
from app.services.redis_service import redis_service

logger = logging.getLogger(__name__)
settings = get_settings()


class FAISSService:
    """Scalable FAISS Indexing Service for enterprise scale speaker identification."""

    def __init__(self, index_dir: str = "./data", index_type: str | None = None):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, os.path.basename(settings.FAISS_INDEX_PATH))
        self.index_type = (index_type or os.environ.get("FAISS_INDEX_TYPE", "flat")).lower()
        self.dimension = settings.EMBEDDING_DIM
        self.m = settings.FAISS_M
        self.nbits = settings.FAISS_NBITS
        
        # Double-buffering thread safety lock
        self._lock = threading.Lock()
        
        # Active in-memory FAISS index and mappings
        self.index: Optional[faiss.Index] = None
        self.index_to_uuid: Dict[int, str] = {}
        self.uuid_to_index: Dict[str, int] = {}
        
        # Initialize
        os.makedirs(self.index_dir, exist_ok=True)
        self.initialize_index()

    def initialize_index(self):
        """Initializes IndexIVFPQ with HNSW quantizer or loads trained index from disk."""
        with self._lock:
            if os.path.exists(self.index_path):
                try:
                    logger.info("Loading trained FAISS IVF-PQ index from disk...")
                    self.index = faiss.read_index(self.index_path)
                    if hasattr(self.index, "nprobe"):
                        self.index.nprobe = settings.FAISS_NPROBE
                    logger.info(f"Loaded FAISS IVF-PQ index containing {self.index.ntotal} vectors.")
                    self._restore_mapping_from_redis_or_file()
                    return
                except Exception as e:
                    logger.error(f"Failed to load FAISS index from disk, initializing IndexFlatIP fallback: {e}")

            # Fallback to empty index
            logger.info("Initializing FAISS %s index.", self.index_type)
            self.index = self._new_empty_index()
            self.index_to_uuid = {}
            self.uuid_to_index = {}

    def _new_empty_index(self) -> faiss.Index:
        """Create an empty FAISS index according to the configured index type."""
        if self.index_type == "hnsw":
            hnsw_m = int(os.environ.get("FAISS_HNSW_M", "32"))
            index = faiss.IndexHNSWFlat(self.dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = int(os.environ.get("FAISS_HNSW_EF_SEARCH", "64"))
                index.hnsw.efConstruction = int(os.environ.get("FAISS_HNSW_EF_CONSTRUCTION", "80"))
            return index

        if self.index_type == "ivfpq":
            nlist = int(os.environ.get("FAISS_IVFPQ_NLIST", "100"))
            quantizer = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                nlist,
                self.m,
                self.nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            if hasattr(index, "nprobe"):
                index.nprobe = settings.FAISS_NPROBE
            return index

        return faiss.IndexFlatIP(self.dimension)

    def _restore_mapping_from_redis_or_file(self):
        """Restore FAISS ID mappings from Redis if available, else local json file."""
        mapping = redis_service.get("faiss_mapping_json")
        if mapping:
            try:
                data = json.loads(mapping)
                self.index_to_uuid = {int(k): v for k, v in data.get("index_to_uuid", {}).items()}
                self.uuid_to_index = data.get("uuid_to_index", {})
                logger.info("Restored FAISS ID mappings from Redis.")
                return
            except Exception as e:
                logger.warning(f"Could not parse Redis FAISS mapping: {e}")

        # Local fallback
        mapping_path = os.path.join(self.index_dir, "faiss_mapping.json")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r") as f:
                    data = json.load(f)
                    self.index_to_uuid = {int(k): v for k, v in data.get("index_to_uuid", {}).items()}
                    self.uuid_to_index = data.get("uuid_to_index", {})
                logger.info("Restored FAISS ID mappings from local JSON file.")
            except Exception as e:
                logger.error(f"Could not load local FAISS mapping JSON: {e}")

    def _normalize_vector(self, vector: List[float]) -> np.ndarray:
        """Converts vector to float32 numpy array and L2-normalizes it."""
        arr = np.array(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(arr)
        return arr

    def save_to_disk(self):
        """Saves current index to disk and persists mapping to Redis."""
        try:
            faiss.write_index(self.index, self.index_path)
            mapping_data = json.dumps({
                "index_to_uuid": self.index_to_uuid,
                "uuid_to_index": self.uuid_to_index,
            })
            redis_service.set("faiss_mapping_json", mapping_data)
            
            mapping_path = os.path.join(self.index_dir, "faiss_mapping.json")
            with open(mapping_path, "w") as f:
                f.write(mapping_data)
            logger.info("Saved FAISS index and mappings atomically.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def add_or_update(self, user_id: uuid.UUID, embedding: List[float]):
        """Incremental add/update vector in FAISS with double-buffering lock."""
        user_str = str(user_id)
        norm_emb = self._normalize_vector(embedding)
        
        with self._lock:
            if user_str in self.uuid_to_index:
                self._rebuild_in_memory_without_save(user_str, norm_emb)
            else:
                idx_id = self.index.ntotal
                self.index.add(norm_emb)
                self.index_to_uuid[idx_id] = user_str
                self.uuid_to_index[user_str] = idx_id
                
            self.save_to_disk()

    def _rebuild_in_memory_without_save(self, update_user_str: str, new_emb: np.ndarray):
        """Rebuilds the FAISS index in memory during single vector updates."""
        new_index = self._new_empty_index()
        if hasattr(new_index, "is_trained") and not new_index.is_trained:
            existing_vectors = [
                self.index.reconstruct(idx).reshape(1, -1)
                for idx in list(self.uuid_to_index.values())
            ]
            training_vectors = np.vstack(existing_vectors + [new_emb])
            try:
                new_index.train(training_vectors)
            except Exception:
                logger.warning("Falling back to IndexFlatIP for incremental update; configured index could not train.")
                new_index = faiss.IndexFlatIP(self.dimension)
        new_index_to_uuid = {}
        new_uuid_to_index = {}
        
        for u_str, idx in list(self.uuid_to_index.items()):
            if u_str == update_user_str:
                emb = new_emb
            else:
                emb = self.index.reconstruct(idx).reshape(1, -1)
            
            new_idx = new_index.ntotal
            new_index.add(emb)
            new_index_to_uuid[new_idx] = u_str
            new_uuid_to_index[u_str] = new_idx
            
        self.index = new_index
        self.index_to_uuid = new_index_to_uuid
        self.uuid_to_index = new_uuid_to_index

    def search(self, embedding: List[float], k: int = 1) -> List[Tuple[uuid.UUID, float]]:
        """
        Searches index for top-k similar voiceprints.
        Returns List of (user_uuid, similarity_score).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
            
        norm_emb = self._normalize_vector(embedding)
        
        with self._lock:
            scores, indices = self.index.search(norm_emb, min(k, self.index.ntotal))
            
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            user_str = self.index_to_uuid.get(int(idx))
            if user_str:
                results.append((uuid.UUID(user_str), float(score)))
                
        return results

    async def sync_with_db(self, db_session):
        """Atomically rebuilds index completely from PostgreSQL using double-buffering lock."""
        logger.info("Starting FAISS IVF-PQ index double-buffer rebuild from PostgreSQL...")
        try:
            result = await db_session.execute(select(Voiceprint))
            voiceprints = result.scalars().all()
            
            new_index = self._new_empty_index()
            new_index_to_uuid = {}
            new_uuid_to_index = {}
            
            embeddings = []
            user_uuids = []
            
            for vp in voiceprints:
                if vp.encrypted_embedding and vp.embedding_nonce:
                    try:
                        from app.services.voiceprint_crypto import VoiceprintCrypto
                        crypto = VoiceprintCrypto.from_env()
                        embedding = crypto.decrypt_embedding(vp.encrypted_embedding, vp.embedding_nonce)
                        biohash = crypto.decrypt_biohash(vp.encrypted_biohash, vp.biohash_nonce)
                        vp.attach_plaintext(embedding, biohash)
                    except Exception as err:
                        logger.warning(f"Failed to decrypt voiceprint for user {vp.user_id}: {err}")
                if vp.embedding is not None:
                    embeddings.append(vp.embedding)
                    user_uuids.append(str(vp.user_id))
                
            if embeddings:
                emb_matrix = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(emb_matrix)
                if hasattr(new_index, "is_trained") and not new_index.is_trained:
                    try:
                        new_index.train(emb_matrix)
                    except Exception:
                        logger.warning("Falling back to IndexFlatIP; configured FAISS index could not train on current corpus.")
                        new_index = faiss.IndexFlatIP(self.dimension)
                new_index.add(emb_matrix)
                
                for i, u_str in enumerate(user_uuids):
                    new_index_to_uuid[i] = u_str
                    new_uuid_to_index[u_str] = i
                    
            # Atomic double-buffer swap
            with self._lock:
                self.index = new_index
                self.index_to_uuid = new_index_to_uuid
                self.uuid_to_index = new_uuid_to_index
                self.save_to_disk()
                
            logger.info(f"FAISS double-buffer index swap complete. Registered {self.index.ntotal} users.")
        except Exception as e:
            logger.error(f"Error during FAISS database sync: {e}")
            raise


# Singleton instance
faiss_service = FAISSService()
