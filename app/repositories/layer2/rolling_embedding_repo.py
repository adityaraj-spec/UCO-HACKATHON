"""
app/repositories/layer2/rolling_embedding_repo.py
"""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.layer2.rolling_embedding import RollingEmbedding


class RollingEmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_user_id(self, user_id: uuid.UUID) -> list[RollingEmbedding]:
        stmt = select(RollingEmbedding).where(
            RollingEmbedding.user_id == user_id,
            RollingEmbedding.is_active == True
        ).order_by(RollingEmbedding.pool_position.asc())
        res = await self.session.execute(stmt)
        return list(res.scalars().all())
