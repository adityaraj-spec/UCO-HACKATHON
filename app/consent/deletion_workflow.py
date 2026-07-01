"""
app/consent/deletion_workflow.py

Automated biometric data deletion scheduler (DPDP Compliance Purge).

When consent is withdrawn, DPDP Act 2023 requires all biometric identities
and derived secrets to be purged within 7 days.
This workbook scheduler performs safe deletions across:
  1. FAISS HNSW indexes
  2. Redis cache pools
  3. PostgreSQL Anchor and Rolling embeddings tables
  4. User threshold, metadata profiles
"""

from __future__ import annotations

import logging
import uuid

from app.cache.session_cache import SessionCache
from app.search.faiss_index import get_faiss_manager
from app.database.session import AsyncSession

logger = logging.getLogger(__name__)


class BiometricPurgeScheduler:
    """Orchestrates secure deletion of user biometric records upon consent withdrawal."""

    def __init__(self) -> None:
        self.faiss_manager = get_faiss_manager()
        self.session_cache = SessionCache()

    async def execute_purge(self, db: AsyncSession, user_id: uuid.UUID) -> bool:
        """
        Perform a cryptographic database clean of all user records.

        Args:
            db: Active database transaction session
            user_id: The target user to be purged

        Returns:
            True if all records deleted successfully
        """
        logger.info("Executing biometric purge for user %s", user_id)

        try:
            # 1. Evict from local/cluster FAISS vectors
            self.faiss_manager.remove_vector(str(user_id))

            # 2. Invalidate Redis rolling template entries
            await self.session_cache.invalidate_embeddings_cache(user_id)

            # 3. Clean SQL tables: Anchor, Rolling, Thresholds, Voice Quality Metadata
            # Import models dynamically to avoid import circular loops
            from app.models.layer2.anchor_embedding import AnchorEmbedding
            from app.models.layer2.rolling_embedding import RollingEmbedding
            from app.models.layer2.user_threshold import UserThreshold
            from app.models.layer2.voice_metadata import VoiceMetadata
            from sqlalchemy import delete

            # Delete Anchor
            await db.execute(delete(AnchorEmbedding).where(AnchorEmbedding.user_id == user_id))
            # Delete Rolling Pool
            await db.execute(delete(RollingEmbedding).where(RollingEmbedding.user_id == user_id))
            # Delete thresholds
            await db.execute(delete(UserThreshold).where(UserThreshold.user_id == user_id))
            # Delete metadata profile
            await db.execute(delete(VoiceMetadata).where(VoiceMetadata.user_id == user_id))
            # Layer 1 Voiceprint purge is handled separately via Layer 1 service (not imported here)

            logger.info("Biometric purge logic completed successfully for user: %s", user_id)
            return True

        except Exception as e:
            logger.exception("Failed to execute biometric purge for user %s: %s", user_id, e)
            return False


# Module-level singleton
_purge_scheduler: BiometricPurgeScheduler | None = None


def get_purge_scheduler() -> BiometricPurgeScheduler:
    """Return singleton BiometricPurgeScheduler."""
    global _purge_scheduler
    if _purge_scheduler is None:
        _purge_scheduler = BiometricPurgeScheduler()
    return _purge_scheduler
