"""
app/repositories/layer2/auth_history_repo.py
"""

from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.layer2.auth_history import AuthHistory


class AuthHistoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        user_id: uuid.UUID,
        session_id: str,
        phrase_id: str,
        attempt_number: int,
        liveness_score: float,
        is_liveness_passed: bool,
        similarity_score: float,
        is_similarity_passed: bool,
        final_decision: str,
        risk_score: float,
        risk_level: str,
        failure_reason: str | None = None,
        ip_device_trusted: bool = True,
    ) -> AuthHistory:
        hist = AuthHistory(
            user_id=user_id,
            session_id=session_id,
            challenge_phrase_id=phrase_id,
            auth_result=final_decision,
            weighted_score=similarity_score,
            anchor_similarity_score=similarity_score,
            antispoof_confidence=liveness_score,
            antispoof_result="PASS" if is_liveness_passed else "FAIL",
            liveness_verified=is_liveness_passed,
            fraud_risk_score=risk_score,
            risk_level=risk_level,
            error_message=failure_reason,
        )
        self.session.add(hist)
        await self.session.flush()
        return hist

    async def get_failures_count_24h(self, user_id: uuid.UUID) -> int:
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        stmt = select(AuthHistory).where(
            AuthHistory.user_id == user_id,
            AuthHistory.auth_result != "PASS",
            AuthHistory.created_at >= since
        )
        res = await self.session.execute(stmt)
        return len(res.scalars().all())

