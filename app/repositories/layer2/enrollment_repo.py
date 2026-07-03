"""
app/repositories/layer2/enrollment_repo.py

Fixed to match the actual EnrollmentSession model columns:
  - samples_received (not samples_submitted)
  - samples_accepted (not samples_submitted)
  - session_token (required, unique)
  - expires_at (required)
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.layer2.enrollment import EnrollmentSession


class EnrollmentSessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, session_id: uuid.UUID) -> EnrollmentSession | None:
        return await self.session.get(EnrollmentSession, session_id)

    async def get_active_session(self, user_id: uuid.UUID) -> EnrollmentSession | None:
        stmt = (
            select(EnrollmentSession)
            .where(
                EnrollmentSession.user_id == user_id,
                EnrollmentSession.status == "IN_PROGRESS",
            )
            .order_by(EnrollmentSession.created_at.desc())
            .limit(1)
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def create(
        self, user_id: uuid.UUID, total_samples_required: int = 5
    ) -> EnrollmentSession:
        now = datetime.now(timezone.utc)
        sess = EnrollmentSession(
            user_id=user_id,
            status="IN_PROGRESS",
            session_token=secrets.token_urlsafe(48),  # required unique field
            samples_required=total_samples_required,
            samples_received=0,
            samples_accepted=0,
            expires_at=now + timedelta(seconds=600),  # 10-minute session window
        )
        self.session.add(sess)
        await self.session.flush()
        return sess
