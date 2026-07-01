"""
app/repositories/layer2/emergency_repo.py
"""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.layer2.emergency_contact import EmergencyContact, EmergencyAccessEvent


class EmergencyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_contacts(self, user_id: uuid.UUID) -> list[EmergencyContact]:
        stmt = select(EmergencyContact).where(
            EmergencyContact.user_id == user_id,
            EmergencyContact.is_active == True
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def get_active_event(self, user_id: uuid.UUID) -> EmergencyAccessEvent | None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        stmt = select(EmergencyAccessEvent).where(
            EmergencyAccessEvent.user_id == user_id,
            EmergencyAccessEvent.status == "ACTIVE",
            EmergencyAccessEvent.expires_at > now
        ).order_by(EmergencyAccessEvent.activated_at.desc()).limit(1)
        res = await self.session.execute(stmt)
        return res.scalars().first()
