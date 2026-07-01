"""
app/repositories/layer2/consent_repo.py
"""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.layer2.consent import Consent


class ConsentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_user_id(self, user_id: uuid.UUID) -> Consent | None:
        stmt = select(Consent).where(
            Consent.user_id == user_id,
            Consent.is_active == True
        ).order_by(Consent.created_at.desc()).limit(1)
        res = await self.session.execute(stmt)
        return res.scalars().first()
