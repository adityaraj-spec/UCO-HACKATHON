"""
app/services/biohash_service.py

BioHashing service for generating cancellable binary biometric tokens from continuous embeddings.
Transforms a 192-dimensional vector into a 256-bit BioHash token via orthogonal projection.
"""

import hashlib
import numpy as np
from app.core.config import get_settings

settings = get_settings()


class BioHashService:
    """Encapsulates BioHash generation, projection matrix creation, and Hamming similarity mapping."""

    def __init__(self, seed_key: str | None = None, biohash_dim: int = 256, embedding_dim: int = 192):
        self.seed_key = seed_key or settings.BIOHASH_SEED_KEY
        self.biohash_dim = biohash_dim
        self.embedding_dim = embedding_dim
        self.projection_matrix = self._generate_projection_matrix()

    def _generate_projection_matrix(self) -> np.ndarray:
        """
        Generate a reproducible pseudo-random projection matrix W (256 x 192).
        Uses SHA-256 seed derived from seed_key.
        """
        seed = int(hashlib.sha256(self.seed_key.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        mat = rng.randn(self.biohash_dim, self.embedding_dim).astype(np.float32)
        # Normalise rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-6)

    def compute_biohash(self, embedding: list[float] | np.ndarray) -> str:
        """
        Transform a 192-dim vector into a 256-bit binary BioHash string of '0's and '1's.
        b = sign(W . v)
        """
        arr = np.asarray(embedding, dtype=np.float32)
        projected = np.dot(self.projection_matrix, arr)
        binary_bits = (projected >= 0).astype(int)
        return "".join(map(str, binary_bits))

    def hamming_distance(self, biohash1: str, biohash2: str) -> float:
        """Compute normalised Hamming distance between two 256-bit binary strings (range [0.0, 1.0])."""
        if len(biohash1) != len(biohash2):
            raise ValueError("BioHash tokens must be of equal length")
        mismatches = sum(c1 != c2 for c1, c2 in zip(biohash1, biohash2))
        return mismatches / len(biohash1)

    def biohash_similarity(self, biohash1: str, biohash2: str) -> float:
        """
        Map normalised Hamming distance to equivalent Cosine Similarity score.
        sim = cos(pi * HammingDistance)
        """
        hd = self.hamming_distance(biohash1, biohash2)
        return float(np.cos(np.pi * hd))
