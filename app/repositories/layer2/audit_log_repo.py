"""
app/repositories/layer2/audit_log_repo.py
"""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.layer2.audit_log import AuditLog


class AuditLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_user_id(self, user_id: uuid.UUID) -> list[AuditLog]:
        stmt = select(AuditLog).where(AuditLog.user_id == user_id).order_by(AuditLog.created_at.desc())
        res = await self.session.execute(stmt)
        return list(res.scalars().all())
