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
from datetime import datetime, timedelta
import jwt
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
from app.cache.session_cache import SessionCache
# Middleware
from app.middleware.rbac import allow_customer, allow_admin_only, require_internal_service_token
from app.middleware.rate_limiter import limiter
from app.middleware.jwt_auth import get_current_user_id, get_current_user_claims
from app.models.layer2.audit_log import AuditLog

from app.core.config import get_settings
from app.audit.event_types import AuditEventType

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


# ------------------------------------------------------------------ #
# Consent Management                                                #
# ------------------------------------------------------------------ #

@router.post("/consent/grant", tags=["Consent"])
@limiter.limit("5/minute")
async def grant_consent(
    request: Request,
    consent_type: str = Form("EXPLICIT_OPT_IN"),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Grant customer biometric opt-in consent for voiceprint storage."""
    service = get_consent_service()
    audit = get_audit_vault()
    ip_addr = request.client.host if request.client else "127.0.0.1"
    u_agent = request.headers.get("user-agent", "unknown")

    consent = await service.record_consent_grant(db, user_id, ip_addr, u_agent, consent_type)
    await audit.log_event(
        db=db,
        event_type=AuditEventType.CONSENT_GRANT,
        user_id=user_id,
        actor="CUSTOMER",
        ip_address=ip_addr,
        details="Biometric consent granted.",
        payload={"consent_id": str(consent.id), "consent_type": consent_type},
    )
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
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Withdraw storage consent. Immediately schedules vector and cache purge."""
    service = get_consent_service()
    purge_scheduler = get_purge_scheduler()
    audit = get_audit_vault()
    ip_addr = request.client.host if request.client else "127.0.0.1"
    u_agent = request.headers.get("user-agent", "unknown")

    revoked = await service.record_consent_withdrawal(db, user_id, ip_addr, u_agent)
    if not revoked:
        raise HTTPException(status_code=404, detail="No active consent found to withdraw.")

    # Execute purge synchronously for immediate security response
    purged = await purge_scheduler.execute_purge(db, user_id)
    await audit.log_event(
        db=db,
        event_type=AuditEventType.CONSENT_WITHDRAW,
        user_id=user_id,
        actor="CUSTOMER",
        ip_address=ip_addr,
        details="Biometric consent withdrawn and purge workflow executed.",
        payload={"purged": purged},
    )
    await db.commit()

    return {
        "success": True,
        "purged": purged,
        "message": "Biometric consent withdrawn. Voice profiles successfully purged from system storage."
    }


@router.get("/consent/status", tags=["Consent"])
@limiter.limit("30/minute")
async def get_consent_status(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user_id),
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
    user_id: uuid.UUID = Depends(get_current_user_id),
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
            "samples_submitted": sess.samples_received,
            "samples_required": sess.samples_required,
            "status": sess.status,
        }
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))


@router.post("/voice/enroll/sample", tags=["Enrollment"])
@limiter.limit("30/minute")
async def submit_enrollment_sample(
    request: Request,
    session_id: uuid.UUID = Form(...),
    audio: UploadFile = File(...),
    user_id: uuid.UUID = Depends(get_current_user_id),
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
    session_id: uuid.UUID = Form(...),
    user_id: uuid.UUID = Depends(get_current_user_id),
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
    session_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Request a random challenge phrase and receive a JWT-embedded validation code."""
    service = get_challenge_service()
    phrase_id, phrase_text, token = service.generate_challenge(user_id, session_id)
    try:
        claims = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        await SessionCache().register_challenge_token(claims["jti"], phrase_id, user_id)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=500, detail=f"Challenge token creation failed: {exc}") from exc

    ip_addr = request.client.host if request.client else "127.0.0.1"
    await get_audit_vault().log_event(
        db=db,
        event_type=AuditEventType.AUTH_CHALLENGE_ISSUED,
        user_id=user_id,
        actor="CUSTOMER",
        ip_address=ip_addr,
        details="Voice challenge phrase issued.",
        payload={"session_id": session_id, "phrase_id": phrase_id},
    )
    await db.commit()
    return {
        "phrase_id": phrase_id,
        "phrase_text": phrase_text,
        "challenge_token": token,
    }


@router.post("/voice/authenticate", tags=["Verification"])
@limiter.limit("30/minute")
async def authenticate_voice(
    request: Request,
    session_id: str = Form(...),
    challenge_token: str = Form(...),
    device_fingerprint: str = Form(...),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    audio: UploadFile = File(...),
    user_id: uuid.UUID = Depends(get_current_user_id),
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
        raise HTTPException(status_code=500, detail=f"Authentication transaction error: {e}")


@router.post("/voice/authenticate/direct", tags=["Verification"])
@limiter.limit("30/minute")
async def authenticate_voice_direct(
    request: Request,
    device_fingerprint: str = Form(...),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    audio: UploadFile = File(...),
    _internal_authorized: bool = Depends(require_internal_service_token),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Directly authenticate voice embedding without challenge tokens."""
    service = AuthService(db)
    ip_addr = request.client.host if request.client else "127.0.0.1"

    try:
        res = await service.authenticate_direct(
            user_id=user_id,
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
        logger.exception("Voice authentication direct critical error: %s", e)
        raise HTTPException(status_code=500, detail=f"Authentication direct transaction error: {e}")



# ------------------------------------------------------------------ #
# Emergency Access Overrides                                         #
# ------------------------------------------------------------------ #

@router.post("/emergency/contact", tags=["Emergency Access"])
@limiter.limit("5/minute")
async def register_emergency_contact(
    request: Request,
    contact_name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    mpin: str = Form(...),
    primary_otp: str = Form(...),
    scope: EmergencyAccessScope = Form(EmergencyAccessScope.READ_ONLY),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Register verified contact trustee authorized for emergency check overrides."""
    from app.repositories.user_repository import UserRepository
    from app.utils.mpin_hasher import verify_mpin
    from app.emergency.otp_interface import get_otp_provider

    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found.")

    # 1. Step-up: Verify current MPIN
    if not verify_mpin(mpin, user.mpin_hash):
        raise HTTPException(status_code=401, detail="Invalid primary account holder MPIN.")

    # 2. Step-up: Verify OTP sent to primary holder's phone
    otp_prov = get_otp_provider()
    otp_valid = await otp_prov.verify_otp(user.phone_number, primary_otp)
    if not otp_valid:
        raise HTTPException(status_code=400, detail="Invalid OTP code.")

    service = get_emergency_contact_service()
    try:
        contact = await service.register_contact(db, user_id, contact_name, phone, email, scope)
        # Send SMS trigger immediately for mock testing
        otp_ref = await service.initiate_verification_otp(contact)
        await get_audit_vault().log_event(
            db=db,
            event_type=AuditEventType.EMERGENCY_CONTACT_REGISTER,
            user_id=user_id,
            actor="CUSTOMER",
            ip_address=request.client.host if request.client else "127.0.0.1",
            details=f"Emergency contact registered with scope {scope}.",
            payload={"contact_id": str(contact.id), "scope": str(scope)},
        )
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
    trustee_phone: str = Form(...),
    otp_code: str = Form(None),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Request and verify OTP validation verification to spawn emergency sessions."""
    service = get_emergency_activation_service()
    contact = await service.find_contact_by_phone(db, user_id, trustee_phone)
    if not contact:
        raise HTTPException(status_code=403, detail="Unregistered or unverified trustee credentials.")

    # Enforce activation cooldown delay (48 hours)
    from datetime import timezone
    cooling_expiry = contact.created_at + timedelta(hours=settings.EMERGENCY_CONTACT_ACTIVATION_DELAY_HOURS)
    if datetime.now(timezone.utc) < cooling_expiry:
        raise HTTPException(
            status_code=400,
            detail=f"Emergency contact is in cooling-off period. Activation not permitted until {cooling_expiry.isoformat()}."
        )

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


@router.post("/emergency/legal-guardian/request", tags=["Emergency Access"])
@limiter.limit("5/minute")
async def create_legal_guardian_request(
    request: Request,
    user_id: uuid.UUID = Form(...),
    pdf: UploadFile = File(...),
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """
    Tier 4: Back-office workflow. Create a manual activation request with an uploaded PDF document.
    Must be called by a staff member.
    """
    role = claims.get("role", "CUSTOMER")
    if role not in ["ADMIN", "AGENT", "STAFF", "branch_manager", "compliance_officer"]:
        raise HTTPException(status_code=403, detail="Staff privilege required.")

    # Check if the PDF has a valid PDF extension
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are permitted.")

    from pathlib import Path
    upload_dir = Path("data/legal_guardians")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4().hex}_{pdf.filename}"

    try:
        content = await pdf.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF upload: {e}")

    from app.models.layer2.emergency_contact import LegalGuardianRequest
    req = LegalGuardianRequest(
        user_id=user_id,
        pdf_path=str(file_path),
        status="PENDING_REVIEW"
    )
    db.add(req)
    await db.commit()
    await db.refresh(req)

    return {
        "success": True,
        "request_id": str(req.id),
        "status": req.status,
        "pdf_path": req.pdf_path,
        "message": "Court Order / Legal Guardian request submitted successfully for dual-approval."
    }


@router.post("/emergency/legal-guardian/approve", tags=["Emergency Access"])
@limiter.limit("5/minute")
async def approve_legal_guardian_request(
    request: Request,
    request_id: uuid.UUID = Form(...),
    claims: dict = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_db)
):
    """
    Tier 4 Dual-Approval endpoint. Enforces branch_manager and compliance_officer signatures.
    """
    role = claims.get("role")
    actor_id = claims.get("sub")
    if role not in ["branch_manager", "compliance_officer"]:
        raise HTTPException(
            status_code=403,
            detail="Required role branch_manager or compliance_officer missing."
        )

    from app.models.layer2.emergency_contact import LegalGuardianRequest, EmergencyContact, EmergencyAccessEvent
    from app.encryption.aes_gcm import AESGCM256
    from app.encryption.key_manager import get_key_manager
    from datetime import datetime, timezone
    import secrets

    req = await db.get(LegalGuardianRequest, request_id)
    if not req:
        raise HTTPException(status_code=404, detail="Legal guardian request not found.")

    if req.status in ["APPROVED", "REJECTED"]:
        raise HTTPException(status_code=400, detail=f"Request already processed. Status: {req.status}")

    if role == "branch_manager":
        req.branch_manager_approved = True
        req.branch_manager_id = str(actor_id)
    elif role == "compliance_officer":
        req.compliance_officer_approved = True
        req.compliance_officer_id = str(actor_id)

    # Check if dual approval criteria is met
    if req.branch_manager_approved and req.compliance_officer_approved:
        if req.branch_manager_id == req.compliance_officer_id:
            # Save progress up to here but error out
            db.add(req)
            await db.commit()
            raise HTTPException(
                status_code=400,
                detail="Branch manager and compliance officer signatures must be from distinct individuals."
            )

        req.status = "APPROVED"

        # Provision the special EmergencyContact and EmergencyAccessEvent
        key_id = settings.HSM_MASTER_KEY_ID
        user_key = get_key_manager().get_key_for_user(str(req.user_id), key_id)
        encryptor = AESGCM256(user_key)

        phone_enc = encryptor.encrypt(b"0000000000", associated_data=str(req.user_id).encode())
        email_enc = encryptor.encrypt(b"legal@guardian.com", associated_data=str(req.user_id).encode())

        contact = EmergencyContact(
            user_id=req.user_id,
            contact_name="Court-Appointed Legal Guardian",
            encrypted_phone=phone_enc.ciphertext,
            phone_nonce=phone_enc.nonce,
            encrypted_email=email_enc.ciphertext,
            email_nonce=email_enc.nonce,
            access_scope="FULL_TRANSACTION",
            key_id=key_id,
            is_active=True,
            is_verified=True,
        )
        db.add(contact)
        await db.flush()

        now = datetime.now(timezone.utc)
        from datetime import timedelta
        expires = now + timedelta(hours=settings.EMERGENCY_ACCESS_TTL_HOURS)
        ip_addr = request.client.host if request.client else "127.0.0.1"

        access_event = EmergencyAccessEvent(
            emergency_contact_id=contact.id,
            user_id=req.user_id,
            activation_reason="Court Order / Legal Guardian authorized via branch manager and compliance officer approval.",
            verification_method_used="COURT_ORDER",
            access_token_jti=secrets.token_hex(16),
            access_scope_granted="FULL_TRANSACTION",
            ip_address=ip_addr,
            status="ACTIVE",
            activated_at=now,
            expires_at=expires,
        )
        db.add(access_event)

        # Log event in Immutable Audit Vault
        from app.audit.vault import get_audit_vault
        audit = get_audit_vault()
        await audit.log_event(
            db=db,
            event_type=AuditEventType.EMERGENCY_ACTIVATE,
            user_id=req.user_id,
            actor="SYSTEM_DUAL_APPROVAL",
            ip_address=ip_addr,
            details=f"Court Order / Legal Guardian override activated after double approval. Request: {req.id}.",
            payload={
                "request_id": str(req.id),
                "branch_manager_id": req.branch_manager_id,
                "compliance_officer_id": req.compliance_officer_id,
                "expires_at": expires.isoformat()
            }
        )

    await db.commit()

    return {
        "success": True,
        "request_id": str(req.id),
        "status": req.status,
        "branch_manager_approved": req.branch_manager_approved,
        "compliance_officer_approved": req.compliance_officer_approved,
        "message": f"Approval signature recorded for role: {role}."
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
