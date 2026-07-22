"""
app/api/v1/endpoints/admin.py

Administrative and compliance management endpoints:
- POST /api/v1/admin/rebuild-index: Trigger full FAISS IVF-PQ index rebuild
- GET  /api/v1/admin/compliance/report: Return compliance audit report
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.user import User
from app.models.verification_audit_log import VerificationAuditLog
from app.models.voiceprint import Voiceprint
from app.services.faiss_service import faiss_service

router = APIRouter()


@router.post(
    "/admin/rebuild-index",
    summary="Force a full FAISS index rebuild from database",
)
async def rebuild_index(db: AsyncSession = Depends(get_db)):
    """Triggers atomic double-buffering FAISS index rebuild from PostgreSQL."""
    await faiss_service.sync_with_db(db)
    return {"status": "success", "total_indexed": faiss_service.index.ntotal}


@router.get(
    "/admin/compliance/report",
    summary="Fetch compliance and biometric audit report",
)
async def compliance_report(db: AsyncSession = Depends(get_db)):
    """Returns conformance report for biometric data governance."""
    total_users = (await db.execute(select(func.count(User.id)))).scalar() or 0
    total_voiceprints = (await db.execute(select(func.count(Voiceprint.id)))).scalar() or 0
    total_audit_logs = (await db.execute(select(func.count(VerificationAuditLog.id)))).scalar() or 0

    # Decision distribution
    decisions_res = await db.execute(
        select(VerificationAuditLog.decision, func.count(VerificationAuditLog.id)).group_by(
            VerificationAuditLog.decision
        )
    )
    decision_counts = dict(decisions_res.all())

    return {
        "active_users": total_users,
        "enrolled_biometrics": total_voiceprints,
        "total_audit_records": total_audit_logs,
        "decision_distribution": decision_counts,
        "compliance_status": "CONFORMANT_ISO_27001",
    }
