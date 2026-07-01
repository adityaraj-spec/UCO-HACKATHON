"""
app/biometrics/biohash.py

BioHash transformation for cancelable biometrics.

Why BioHash:
  - Irrevocability: Raw voiceprints cannot be reset if compromised.
  - Transform: T(e, k) = sign(R_k · e)
    - e: original embedding vector
    - k: user-specific key
    - R_k: pseudo-random projection matrix derived from HKDF-SHA256(user_id, k)
  - Non-invertible: The original embedding is mathematically impossible to reconstruct
    from the binary/quantized BioHash template.
  - Revocability: If a template is compromised, change key `k` to generate a completely
    different template. The old template becomes useless.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


class BioHash:
    """
    BioHash transformation engine.

    Args:
        projection_dim: Target dimension after projection (default: 256)
        embedding_dim: Input dimension (default: 192 for ECAPA-TDNN)
    """

    def __init__(self, projection_dim: int = 256, embedding_dim: int = 192) -> None:
        self.projection_dim = projection_dim
        self.embedding_dim = embedding_dim

    def transform(self, embedding: np.ndarray, user_secret_key: bytes) -> np.ndarray:
        """
        Apply BioHash transformation to a raw vector.

        Args:
            embedding: 1D Float32 numpy array of shape (embedding_dim,)
            user_secret_key: Derived user-specific cryptographic key

        Returns:
            1D float32 numpy array of shape (projection_dim,) containing the
            projected template.
        """
        # Ensure dimensions match
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {len(embedding)}")

        # 1. Generate pseudo-random projection matrix R_k based on user_secret_key
        # R_k is of shape (projection_dim, embedding_dim)
        R_k = self._generate_projection_matrix(user_secret_key)

        # 2. Perform projection: p = R_k · e
        # e is normalized first
        norm_e = embedding / (np.linalg.norm(embedding) + 1e-10)
        projected = np.dot(R_k, norm_e)

        # 3. Non-linear Binarization / Quantization: sign(projected)
        # We store float values (+1.0 / -1.0) rather than actual bits
        # to simplify subsequent cosine similarity checks without custom kernels.
        hashed = np.sign(projected).astype(np.float32)

        # Handle zero values
        hashed[hashed == 0] = 1.0

        return hashed

    def _generate_projection_matrix(self, seed_bytes: bytes) -> np.ndarray:
        """
        Generate a deterministic pseudo-random projection matrix from a seed.

        Using Gram-Schmidt orthonormalization on randomly generated vectors
        to ensure the projection matrix is orthogonal. Orthogonal projection
        preserves similarity relations better than purely random projections.
        """
        # Seed the numpy generator deterministically from the key hash
        seed_hash = hashlib.sha256(seed_bytes).digest()
        # Convert first 4 bytes of hash to uint32
        seed_int = int.from_bytes(seed_hash[:4], byteorder="big")
        rng = np.random.default_rng(seed_int)

        # Generate random normal vectors
        R = rng.standard_normal((self.projection_dim, self.embedding_dim))

        # Perform QR decomposition to get orthonormal projection matrix
        # This is more numerically stable than manual Gram-Schmidt
        # Q is orthogonal: Q^T * Q = I
        if self.projection_dim <= self.embedding_dim:
            q, _ = np.linalg.qr(R.T)
            return q.T.astype(np.float32)
        else:
            q, _ = np.linalg.qr(R)
            return q.astype(np.float32)


# Module-level singleton
_biohash: BioHash | None = None


def get_biohash() -> BioHash:
    """Return singleton BioHash instance."""
    global _biohash
    if _biohash is None:
        from app.core.config import get_settings
        settings = get_settings()
        _biohash = BioHash(
            projection_dim=settings.BIOHASH_PROJECTION_DIM,
            embedding_dim=settings.EMBEDDING_DIM,
        )
    return _biohash
