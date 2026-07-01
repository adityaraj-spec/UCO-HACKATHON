"""
app/embeddings/weighted_scorer.py

Ensemble Weighted Scoring Engine.

To prevent template aging/drift while maintaining biometric baseline integrity:
  - baseline: 1 anchor embedding (original enrollment baseline)
  - rolling: up to 10 recent successful auth embeddings

Authenticating score is computed via a weighted average:
  score = 0.4 × cos_sim(auth, anchor) + 0.6 × mean(cos_sim(auth, rolling[i]))

If no rolling embeddings are present, score defaults to cos_sim(auth, anchor).
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class WeightedScorer:
    """Calculates the ensemble matching score between auth and stored templates."""

    def __init__(self, anchor_weight: float = 0.4, rolling_weight: float = 0.6) -> None:
        self.anchor_weight = anchor_weight
        self.rolling_weight = rolling_weight

    def compute_score(
        self,
        auth_template: np.ndarray,
        anchor_template: np.ndarray,
        rolling_templates: list[np.ndarray],
    ) -> tuple[float, float, float]:
        """
        Compute the weighted average similarity score.

        Args:
            auth_template: Plaintext BioHash template of live attempt (quantized/projected)
            anchor_template: Decrypted BioHash anchor template
            rolling_templates: List of decrypted BioHash rolling pool templates

        Returns:
            Tuple of:
              - weighted_score (combined decision metric)
              - anchor_similarity
              - rolling_average_similarity
        """
        # Norms are expected to be checked, but BioHash output templates
        # consisting of +1.0/-1.0 floats benefit from standard cosine similarity:
        # cos_sim(A, B) = dot(A, B) / (||A|| * ||B||)

        # 1. Similarity vs. Anchor
        anchor_sim = self._cosine_similarity(auth_template, anchor_template)

        # 2. Similarity vs. Rolling templates
        if not rolling_templates:
            # If rolling pool is empty, decision relies entirely on the anchor
            logger.debug("Rolling pool is empty, falling back to anchor-only score")
            return anchor_sim, anchor_sim, 0.0

        rolling_sims = [
            self._cosine_similarity(auth_template, r) for r in rolling_templates
        ]
        rolling_avg = float(np.mean(rolling_sims))

        # 3. Weighted Average
        weighted = (self.anchor_weight * anchor_sim) + (self.rolling_weight * rolling_avg)

        logger.debug(
            "Ensemble score: weighted=%.3f, anchor_sim=%.3f, rolling_avg=%.3f (samples=%d)",
            weighted, anchor_sim, rolling_avg, len(rolling_templates),
        )

        return weighted, anchor_sim, rolling_avg

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1D arrays."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


# Module-level singleton
_scorer: WeightedScorer | None = None


def get_weighted_scorer() -> WeightedScorer:
    """Return singleton WeightedScorer."""
    global _scorer
    if _scorer is None:
        _scorer = WeightedScorer()
    return _scorer
