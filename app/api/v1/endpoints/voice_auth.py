"""
app/api/v1/endpoints/voice_auth.py

FastAPI endpoint routers for the Layer 2 Voice Biometric system.

Exposes:
  - DPDP Consent grant, withdraw, status
  - Session-based voice enrollment (session, sample, complete)
  - OTP challenge phrase request
  - Multimodal speaker authentication
  - Emergency contact register and trustee activation
  - Chained immutable audit log visualizer
  - Subsystem local health status reporting
"""

from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
# Repos and services
from app.services.enrollment_service import EnrollmentService
from app.services.auth_service import AuthService
from app.consent.consent_service import get_consent_service
from app.consent.deletion_workflow import get_purge_scheduler
from app.emergency.contact_service import get_emergency_contact_service
from app.emergency.activation_service import get_emergency_activation_service
from app.emergency.access_scope import EmergencyAccessScope
from app.audit.vault import get_audit_vault
from app.antispoof.challenge_service import get_challenge_service
# Middleware
from app.middleware.rbac import allow_customer, allow_admin_only
from app.middleware.rate_limiter import limiter
from app.models.layer2.audit_log import AuditLog

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------ #
# Consent Management                                                #
# ------------------------------------------------------------------ #

@router.post("/consent/grant", tags=["Consent"])
@limiter.limit("5/minute")
async def grant_consent(
    request: Request,
    user_id: uuid.UUID = Form(...),
    consent_type: str = Form("EXPLICIT_OPT_IN"),
    db: AsyncSession = Depends(get_db)
):
    """Grant customer biometric opt-in consent for voiceprint storage."""
    service = get_consent_service()
    ip_addr = request.client.host if request.client else "127.0.0.1"
    u_agent = request.headers.get("user-agent", "unknown")

    consent = await service.record_consent_grant(db, user_id, ip_addr, u_agent, consent_type)
    await db.commit()

    return {
        "success": True,
        "consent_id": str(consent.id),
        "consent_token": consent.consent_token,
        "message": "Explicit voice biometric storage consent has been granted and recorded."
    }


@router.post("/consent/withdraw", tags=["Consent"])
@limiter.limit("5/minute")
async def withdraw_consent(
    request: Request,
    user_id: uuid.UUID = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Withdraw storage consent. Immediately schedules vector and cache purge."""
    service = get_consent_service()
    purge_scheduler = get_purge_scheduler()
    ip_addr = request.client.host if request.client else "127.0.0.1"
    u_agent = request.headers.get("user-agent", "unknown")

    revoked = await service.record_consent_withdrawal(db, user_id, ip_addr, u_agent)
    if not revoked:
        raise HTTPException(status_code=404, detail="No active consent found to withdraw.")

    # Execute purge synchronously for immediate security response
    purged = await purge_scheduler.execute_purge(db, user_id)
    await db.commit()

    return {
        "success": True,
        "purged": purged,
        "message": "Biometric consent withdrawn. Voice profiles successfully purged from system storage."
    }


@router.get("/consent/status/{user_id}", tags=["Consent"])
@limiter.limit("30/minute")
async def get_consent_status(
    request: Request,
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve compliance opt-in state for a certain account."""
    service = get_consent_service()
    active = await service.verify_active_consent(db, user_id)
    return {
        "user_id": user_id,
        "has_active_consent": active
    }


# ------------------------------------------------------------------ #
# Voice Enrollment                                                  #
# ------------------------------------------------------------------ #

@router.post("/voice/enroll/session", tags=["Enrollment"])
@limiter.limit("5/minute")
async def initiate_enrollment_session(
    request: Request,
    user_id: uuid.UUID = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Authenticate consent status and start a 5-sample voice registration workbook."""
    service = EnrollmentService(db)
    ip_addr = request.client.host if request.client else "127.0.0.1"
    u_agent = request.headers.get("user-agent", "unknown")

    try:
        sess = await service.create_enrollment_session(user_id, ip_addr, u_agent)
        await db.commit()

        return {
            "success": True,
            "session_id": str(sess.id),
            "samples_submitted": sess.samples_submitted,
            "samples_required": sess.samples_required,
            "status": sess.status,
        }
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))


@router.post("/voice/enroll/sample", tags=["Enrollment"])
@limiter.limit("30/minute")
async def submit_enrollment_sample(
    request: Request,
    user_id: uuid.UUID = Form(...),
    session_id: uuid.UUID = Form(...),
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Submit single utterance WAV recording file (requires 5 distinct uploads)."""
    service = EnrollmentService(db)
    ip_addr = request.client.host if request.client else "127.0.0.1"

    try:
        report = await service.submit_sample(user_id, session_id, audio, ip_addr)
        await db.commit()
        return report
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.post("/voice/enroll/complete", tags=["Enrollment"])
@limiter.limit("5/minute")
async def complete_enrollment(
    request: Request,
    user_id: uuid.UUID = Form(...),
    session_id: uuid.UUID = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Finalize submitted recordings, generate BioHash profile, and release resources."""
    service = EnrollmentService(db)
    ip_addr = request.client.host if request.client else "127.0.0.1"

    try:
        report = await service.finalize_enrollment(user_id, session_id, ip_addr)
        await db.commit()
        return report
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to finalize biometric templates: {e}")


# ------------------------------------------------------------------ #
# Verification & Audio Authentication                               #
# ------------------------------------------------------------------ #

@router.get("/voice/challenge", tags=["Verification"])
@limiter.limit("60/minute")
async def get_challenge(
    request: Request,
    user_id: uuid.UUID,
    session_id: str,
):
    """Request a random challenge phrase and receive a JWT-embedded validation code."""
    service = get_challenge_service()
    phrase_id, phrase_text, token = service.generate_challenge(user_id, session_id)
    return {
        "phrase_id": phrase_id,
        "phrase_text": phrase_text,
        "challenge_token": token,
    }


@router.post("/voice/authenticate", tags=["Verification"])
@limiter.limit("30/minute")
async def authenticate_voice(
    request: Request,
    user_id: uuid.UUID = Form(...),
    session_id: str = Form(...),
    challenge_token: str = Form(...),
    device_fingerprint: str = Form(...),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Authenticate voice template against user ensemble."""
    service = AuthService(db)
    ip_addr = request.client.host if request.client else "127.0.0.1"

    try:
        res = await service.authenticate(
            user_id=user_id,
            session_id=session_id,
            challenge_token=challenge_token,
            audio_file=audio,
            ip_address=ip_addr,
            device_fingerprint=device_fingerprint,
            latitude=latitude,
            longitude=longitude
        )
        await db.commit()
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Voice authentication critical error: %s", e)
        raise HTTPException(status_code=500, detail=f"Authentication transaction error: {e}")


# ------------------------------------------------------------------ #
# Emergency Access Overrides                                         #
# ------------------------------------------------------------------ #

@router.post("/emergency/contact", tags=["Emergency Access"])
@limiter.limit("5/minute")
async def register_emergency_contact(
    request: Request,
    user_id: uuid.UUID = Form(...),
    contact_name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    scope: EmergencyAccessScope = Form(EmergencyAccessScope.READ_ONLY),
    db: AsyncSession = Depends(get_db)
):
    """Register verified contact trustee authorized for emergency check overrides."""
    service = get_emergency_contact_service()
    try:
        contact = await service.register_contact(db, user_id, contact_name, phone, email, scope)
        # Send SMS trigger immediately for mock testing
        otp_ref = await service.initiate_verification_otp(contact)
        await db.commit()

        return {
            "success": True,
            "contact_id": str(contact.id),
            "is_verified": contact.is_verified,
            "verification_otp_reference": otp_ref,
            "message": "Emergency contact saved. Verification SMS OTP triggered."
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.post("/emergency/contact/verify", tags=["Emergency Access"])
@limiter.limit("10/minute")
async def verify_emergency_contact(
    request: Request,
    contact_id: uuid.UUID = Form(...),
    otp_code: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Verify trustee emergency contact via OTP SMS validation check."""
    service = get_emergency_contact_service()
    from sqlalchemy import select
    from app.models.layer2.emergency_contact import EmergencyContact

    contact = await db.get(EmergencyContact, contact_id)
    if not contact:
        raise HTTPException(status_code=404, detail="Emergency contact profile not found.")

    verified = await service.verify_contact_otp(db, contact, otp_code)
    await db.commit()

    if not verified:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP code.")

    return {
        "success": True,
        "is_verified": True,
        "message": "Contact verified and authorized for emergency delegations."
    }


@router.post("/emergency/activate", tags=["Emergency Access"])
@limiter.limit("5/minute")
async def activate_emergency_override(
    request: Request,
    user_id: uuid.UUID = Form(...),
    trustee_phone: str = Form(...),
    otp_code: str = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """Request and verify OTP validation verification to spawn emergency sessions."""
    service = get_emergency_activation_service()
    contact = await service.find_contact_by_phone(db, user_id, trustee_phone)
    if not contact:
        raise HTTPException(status_code=403, detail="Unregistered or unverified trustee credentials.")

    ip_addr = request.client.host if request.client else "127.0.0.1"

    # Step 1: No OTP sent -> Generate and send OTP SMS
    if not otp_code:
        otp = await service.initiate_activation(db, contact)
        return {
            "success": True,
            "otp_sent": True,
            "verification_otp_reference": otp,
            "message": "Activation challenge SMS generated to trustee device."
        }

    # Step 2: OTP code provided -> Validate and activate 72h session
    event = await service.complete_activation(db, contact, otp_code, ip_addr)
    if not event:
        raise HTTPException(status_code=400, detail="Trustee OTP verification failed.")

    await db.commit()
    return {
        "success": True,
        "session_id": str(event.id),
        "access_scope": event.access_scope,
        "expires_at": event.expires_at.isoformat(),
        "message": f"Biometric override activated for 72 hours. Authorized scope: {event.access_scope}."
    }


# ------------------------------------------------------------------ #
# Immutable Audit Logs visualizer                                    #
# ------------------------------------------------------------------ #

@router.get("/audit/user/{user_id}", tags=["Audit Log"], dependencies=[Depends(allow_admin_only)])
@limiter.limit("10/minute")
async def get_user_audit_logs(
    request: Request,
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve full audit histories for an account. Encapsulated DB signatures."""
    from sqlalchemy import select
    from app.models.layer2.audit_log import AuditLog

    stmt = select(AuditLog).where(AuditLog.user_id == user_id).order_by(AuditLog.created_at.desc())
    res = await db.execute(stmt)
    logs = res.scalars().all()

    # Verify blockchain-style integrity signatures before displaying
    vault = get_audit_vault()
    is_intact, verify_report = await vault.verify_chain_integrity(db)

    return {
        "user_id": user_id,
        "integrity_intact": is_intact,
        "integrity_report": verify_report,
        "chain_length": len(logs),
        "logs": [
            {
                "id": str(l.id),
                "event_type": l.event_type,
                "actor": l.actor,
                "ip_address": l.ip_address,
                "details": l.details,
                "hash_signature": l.hash_signature,
                "previous_hash": l.previous_hash,
                "created_at": l.created_at.isoformat()
            }
            for l in logs
        ]
    }


# ------------------------------------------------------------------ #
# Layer 2 Sub-health                                                #
# ------------------------------------------------------------------ #

@router.get("/health/layer2", tags=["Health"])
async def get_layer2_health(
    db: AsyncSession = Depends(get_db)
):
    """Diagnostic check for Layer 2 systems."""
    # Test Redis client
    from app.cache.redis_client import get_redis_client
    redis_status = "UP"
    try:
        redis_client = await get_redis_client()
        await redis_client.get("health-trigger")
    except Exception:
        redis_status = "DOWN (FALLBACK RUNNING)"

    # Test FAISS index manager
    from app.search.faiss_index import get_faiss_manager
    faiss_status = "UP"
    try:
        faiss = get_faiss_manager()
        _ = faiss.index.ntotal
    except Exception:
        faiss_status = "ERROR"

    # Test Audit Trail
    vault = get_audit_vault()
    is_intact, _ = await vault.verify_chain_integrity(db)

    return {
        "status": "UP",
        "redis_cache": redis_status,
        "faiss_index": faiss_status,
        "audit_vault_integrity": "OK" if is_intact else "TAMPERED",
    }
