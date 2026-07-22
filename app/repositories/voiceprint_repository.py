"""
app/repositories/voiceprint_repository.py

Data-access layer for the `voiceprints` table.

Voiceprints store the averaged ECAPA-TDNN embedding for a user as a
pgvector VECTOR column. This repository handles both creation (first-time
enrollment) and updates (re-enrollment, which overwrites the existing
voiceprint).
"""

import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.voiceprint import Voiceprint


class VoiceprintRepository:
    """Encapsulates all database operations for the Voiceprint model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_user_id(self, user_id: uuid.UUID) -> Voiceprint | None:
        """Fetch the voiceprint belonging to a given user, if any."""
        result = await self.session.execute(
            select(Voiceprint).where(Voiceprint.user_id == user_id)
        )
        voiceprint = result.scalar_one_or_none()
        if voiceprint is not None:
            self._attach_plaintext_voiceprint(voiceprint)
        return voiceprint

    async def upsert(
        self,
        user_id: uuid.UUID,
        embedding: np.ndarray,
        recording_count: int,
    ) -> Voiceprint:
        """
        Create a new voiceprint for `user_id`, or overwrite the existing one
        if enrollment is being re-run.

        Args:
            user_id: The owning user's id.
            embedding: 1-D numpy array (the averaged ECAPA-TDNN embedding).
            recording_count: Number of utterances used to build the
                              embedding.

        Returns:
            The persisted Voiceprint instance.
        """
        from app.core.config import get_settings
        import os
        from app.services.biohash_service import BioHashService
        from app.services.voiceprint_crypto import VoiceprintCrypto
        settings = get_settings()
        model_version = os.path.basename(settings.ECAPA_MODEL_SOURCE)
        embedding_list = embedding.astype(float).tolist()
        biohash = BioHashService().compute_biohash(embedding_list)
        crypto = VoiceprintCrypto.from_env()
        encrypted_embedding = crypto.encrypt_embedding(embedding_list)
        encrypted_biohash = crypto.encrypt_biohash(biohash)

        existing = await self.get_by_user_id(user_id)
        if existing is not None:
            existing._raw_embedding_column = None
            existing.encrypted_embedding = encrypted_embedding.ciphertext
            existing.embedding_nonce = encrypted_embedding.nonce
            existing.embedding_key_id = encrypted_embedding.key_id
            existing._raw_biohash_column = None
            existing.encrypted_biohash = encrypted_biohash.ciphertext if encrypted_biohash else None
            existing.biohash_nonce = encrypted_biohash.nonce if encrypted_biohash else None
            existing.biohash_key_id = encrypted_biohash.key_id if encrypted_biohash else None
            existing.recording_count = recording_count
            existing.model_version = model_version
            existing.attach_plaintext(embedding_list, biohash)
            self.session.add(existing)
            await self.session.flush()
            await self.session.refresh(existing)
            existing.attach_plaintext(embedding_list, biohash)
            return existing

        voiceprint = Voiceprint(
            user_id=user_id,
            encrypted_embedding=encrypted_embedding.ciphertext,
            embedding_nonce=encrypted_embedding.nonce,
            embedding_key_id=encrypted_embedding.key_id,
            encrypted_biohash=encrypted_biohash.ciphertext if encrypted_biohash else None,
            biohash_nonce=encrypted_biohash.nonce if encrypted_biohash else None,
            biohash_key_id=encrypted_biohash.key_id if encrypted_biohash else None,
            recording_count=recording_count,
            model_version=model_version,
        )
        voiceprint._raw_embedding_column = None
        voiceprint._raw_biohash_column = None
        voiceprint.attach_plaintext(embedding_list, biohash)
        self.session.add(voiceprint)
        await self.session.flush()
        await self.session.refresh(voiceprint)
        voiceprint.attach_plaintext(embedding_list, biohash)
        return voiceprint

    async def delete_by_user_id(self, user_id: uuid.UUID) -> bool:
        """Delete the voiceprint for a user, if one exists. Returns True if deleted."""
        existing = await self.get_by_user_id(user_id)
        if existing is None:
            return False
        await self.session.delete(existing)
        await self.session.flush()
        return True

    def _attach_plaintext_voiceprint(self, voiceprint: Voiceprint) -> None:
        from app.services.voiceprint_crypto import VoiceprintCrypto
        if voiceprint.encrypted_embedding and voiceprint.embedding_nonce:
            crypto = VoiceprintCrypto.from_env()
            embedding = crypto.decrypt_embedding(
                voiceprint.encrypted_embedding, voiceprint.embedding_nonce
            )
            biohash = crypto.decrypt_biohash(
                voiceprint.encrypted_biohash, voiceprint.biohash_nonce
            )
            voiceprint.attach_plaintext(embedding, biohash)
        else:
            voiceprint.attach_plaintext(
                voiceprint._raw_embedding_column, voiceprint._raw_biohash_column
            )
