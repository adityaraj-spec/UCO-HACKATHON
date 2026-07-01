"""
app/embeddings/rolling_manager.py

Rolling Embedding Pool Manager.

Handles safe updates to the rolling embedding cache and database pool.
Enforces security constraints preventing "drift attacks" (where an attacker
gradually morphs the stored template toward their voice over multiple auths).
"""

from __future__ import annotations

import logging
import uuid
import numpy as np

from app.biometrics.template_vault import get_template_vault
from app.core.config import get_settings
from app.models.layer2.rolling_embedding import RollingEmbedding
from app.database.session import AsyncSession

logger = logging.getLogger(__name__)
settings = get_settings()


class RollingPoolManager:
    """Manages updates, drift detection, and rotation of rolling embeddings."""

    def __init__(self) -> None:
        self.vault = get_template_vault()
        self.max_embeddings = settings.MAX_ROLLING_EMBEDDINGS
        self.drift_threshold = settings.DRIFT_ALERT_DISTANCE

    def detect_drift(self, anchor: np.ndarray, candidate: np.ndarray) -> float:
        """
        Check if a candidate embedding deviates too far from the anchor.

        Returns:
            Cosine distance: 1.0 - cosine_similarity
        """
        dot = np.dot(anchor, candidate)
        norm_a = np.linalg.norm(anchor)
        norm_b = np.linalg.norm(candidate)
        sim = dot / (norm_a * norm_b + 1e-10)
        dist = 1.0 - float(sim)
        return dist

    async def update_pool(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        auth_history_id: uuid.UUID,
        raw_embedding: np.ndarray,
        auth_score: float,
        anchor_template: np.ndarray,
        existing_rolling: list[RollingEmbedding],
    ) -> bool:
        """
        Process and apply a rolling embedding update.

        Saves new template to DB, rotating out the oldest entry (FIFO)
        if pool size exceeds settings.MAX_ROLLING_EMBEDDINGS.

        Args:
            session: SQLAlchemy session
            user_id: The target user
            auth_history_id: ID of the successful auth transaction
            raw_embedding: Plain raw embedding (NOT transformed)
            auth_score: The weighted similarity score achieved in the auth
            anchor_template: Decrypted BioHash anchor (to evaluate drift)
            existing_rolling: Active list of current rolling models from DB

        Returns:
            True if pool was updated successfully
        """
        # 1. Transform raw_embedding to BioHash candidate
        key_id = settings.HSM_MASTER_KEY_ID
        candidate_template = self.vault.generate_challenge_template(user_id, raw_embedding, key_id)

        # 2. Check drift from anchor template
        dist = self.detect_drift(anchor_template, candidate_template)
        if dist > self.drift_threshold:
            logger.warning(
                "Drift detected for user %s: distance=%.3f > threshold=%.3f. Rejecting rolling update.",
                user_id, dist, self.drift_threshold,
            )
            # Drift is logged in audit trail, update rejected
            return False

        # 3. Generate encrypted cancelable template
        enc_blob, nonce, val_hash, key_used = self.vault.generate_template(user_id, raw_embedding)

        # Sort current pool by position to implement FIFO eviction
        existing_rolling.sort(key=lambda x: x.pool_position)

        if len(existing_rolling) >= self.max_embeddings:
            # Re-use the oldest slot (position 0), delete or deactivate it
            oldest = existing_rolling[0]
            logger.info("Evicting oldest rolling embedding at pool position %d", oldest.pool_position)
            await session.delete(oldest)
            
            # Shift remaining pool entries left
            for r in existing_rolling[1:]:
                r.pool_position -= 1

            new_position = self.max_embeddings - 1
        else:
            new_position = len(existing_rolling)

        # Create new rolling embedding model
        new_rolling = RollingEmbedding(
            user_id=user_id,
            auth_history_id=auth_history_id,
            encrypted_embedding=enc_blob,
            encryption_nonce=nonce,
            embedding_hash=val_hash,
            key_id=key_used,
            auth_similarity_score=auth_score,
            cosine_distance_from_anchor=dist,
            pool_position=new_position,
            is_active=True,
        )

        session.add(new_rolling)
        logger.info(
            "Added new rolling embedding for user %s at pool position %d (drift=%.3f)",
            user_id, new_position, dist,
        )
        return True


# Module-level singleton
_pool_manager: RollingPoolManager | None = None


def get_rolling_manager() -> RollingPoolManager:
    """Return singleton RollingPoolManager."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = RollingPoolManager()
    return _pool_manager
