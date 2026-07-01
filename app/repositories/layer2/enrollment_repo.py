"""
app/repositories/layer2/enrollment_repo.py
"""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.layer2.enrollment import EnrollmentSession


class EnrollmentSessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, session_id: uuid.UUID) -> EnrollmentSession | None:
        return await self.session.get(EnrollmentSession, session_id)

    async def get_active_session(self, user_id: uuid.UUID) -> EnrollmentSession | None:
        stmt = select(EnrollmentSession).where(
            EnrollmentSession.user_id == user_id,
            EnrollmentSession.status == "IN_PROGRESS"
        ).order_by(EnrollmentSession.created_at.desc()).limit(1)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def create(self, user_id: uuid.UUID, total_samples_required: int = 5) -> EnrollmentSession:
        sess = EnrollmentSession(
            user_id=user_id,
            status="IN_PROGRESS",
            samples_submitted=0,
            samples_required=total_samples_required,
        )
        self.session.add(sess)
        await self.session.flush()
        return sess
