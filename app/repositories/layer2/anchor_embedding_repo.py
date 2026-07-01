"""
app/repositories/layer2/anchor_embedding_repo.py
"""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.layer2.anchor_embedding import AnchorEmbedding


class AnchorEmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_user_id(self, user_id: uuid.UUID) -> AnchorEmbedding | None:
        stmt = select(AnchorEmbedding).where(
            AnchorEmbedding.user_id == user_id,
            AnchorEmbedding.is_active == True
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def create(
        self,
        user_id: uuid.UUID,
        encrypted_embedding: bytes,
        encryption_nonce: bytes,
        embedding_hash: str,
        key_id: str,
    ) -> AnchorEmbedding:
        anchor = AnchorEmbedding(
            user_id=user_id,
            encrypted_embedding=encrypted_embedding,
            encryption_nonce=encryption_nonce,
            embedding_hash=embedding_hash,
            key_id=key_id,
            is_active=True,
        )
        self.session.add(anchor)
        await self.session.flush()
        return anchor
