"""
app/services/embedding_aligner.py

Linear alignment service mapping embeddings across fine-tuned ECAPA-TDNN model versions.
T in R^(192x192) maps old embedding spaces to new checkpoint space for backward compatibility.
"""

import numpy as np


class EmbeddingAligner:
    """Manages model version compatibility transformations."""

    def __init__(self):
        # Register known alignment matrices: (old_version, new_version) -> T
        self._transformation_registry: dict[tuple[str, str], np.ndarray] = {}

    def register_transformation(self, old_version: str, new_version: str, matrix: np.ndarray) -> None:
        """Register a 192x192 alignment matrix mapping old_version -> new_version."""
        if matrix.shape != (192, 192):
            raise ValueError(f"Transformation matrix must be shape (192, 192), got {matrix.shape}")
        self._transformation_registry[(old_version, new_version)] = matrix

    def align_embedding(self, embedding: list[float], old_version: str, target_version: str) -> tuple[list[float], bool]:
        """
        Align an embedding from old_version to target_version.
        Returns (aligned_embedding, success_bool).
        """
        if old_version == target_version:
            return embedding, True

        key = (old_version, target_version)
        if key in self._transformation_registry:
            matrix = self._transformation_registry[key]
            arr = np.asarray(embedding, dtype=np.float32)
            aligned = np.dot(matrix, arr)
            # L2 normalize
            norm = np.linalg.norm(aligned)
            if norm > 0:
                aligned = aligned / norm
            return aligned.tolist(), True

        # Fallback: Alignment matrix unavailable
        return embedding, False
