"""
app/repositories/layer2/threshold_repo.py
"""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.layer2.user_threshold import UserThreshold


class UserThresholdRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_user_id(self, user_id: uuid.UUID) -> UserThreshold | None:
        stmt = select(UserThreshold).where(UserThreshold.user_id == user_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_or_create(self, user_id: uuid.UUID) -> UserThreshold:
        existing = await self.get_by_user_id(user_id)
        if existing:
            return existing

        threshold = UserThreshold(
            user_id=user_id,
            threshold_baseline=0.65,
            threshold_current=0.65,
            threshold_floor=0.50,
            threshold_ceiling=0.85,
            consecutive_failures=0,
            illness_window_active=False
        )
        self.session.add(threshold)
        await self.session.flush()
        return threshold
