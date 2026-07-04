# PhaseGuard Layer 2 Overview

PhaseGuard Layer 2 is the speaker-verification service for the banking voice-security flow. Customers authenticate with bank-visible identity fields, while the service keeps UUIDs as internal database keys for biometric, consent, audit, and emergency records.

## Status Table

| Feature | Status | Notes |
|---|---|---|
| Banking identity model | IMPLEMENTED | `users` now stores `account_number`, `mpin_hash`, `full_name`, phone, DOB, branch code, account type, KYC status, active status, last login, MPIN lockout, and MPIN rotation state. |
| Internal UUID primary key | IMPLEMENTED | UUID remains the internal PK and FK target for existing Layer 2 tables. |
| Account Number + MPIN login | IMPLEMENTED | `/api/v1/auth/login` verifies bcrypt MPIN hashes and returns a 15-minute JWT with banking claims. |
| Customer registration | IMPLEMENTED | `/api/v1/auth/register` provisions production-shaped bank customers. |
| Profile endpoint | IMPLEMENTED | `/api/v1/auth/me` returns masked phone numbers and account profile fields. |
| MPIN change | IMPLEMENTED | `/api/v1/auth/mpin/change` requires the old MPIN. |
| MPIN brute-force lockout | IMPLEMENTED | Per-account failure count locks at 5 failed attempts for 30 minutes. |
| MPIN rotation flag | IMPLEMENTED | Login marks `mpin_rotation_required` after `MPIN_MAX_AGE_DAYS`. |
| Login replay protection | IMPLEMENTED | Optional login nonce/timestamp is cached and rejected on replay. Streamlit sends a nonce. |
| JWT refresh | IMPLEMENTED | `/api/v1/auth/refresh` issues a fresh access token from a valid JWT. |
| Account freeze/reactivate | IMPLEMENTED | Customer freeze and staff reactivation endpoints exist; login blocks frozen accounts. |
| KYC gating | IMPLEMENTED | `REJECTED` blocks login; `PENDING` allows login but forces `is_enrolled=false` and UI blocks enrollment. |
| Known-device login signal | IMPLEMENTED | Login records hashed device fingerprints and returns `new_device_login`. |
| Banking Streamlit login | IMPLEMENTED | UI uses Account Number + MPIN and stores bearer JWT. No UUID field is shown in the customer flow. |
| Voice/consent/emergency UUID removal | IMPLEMENTED | Customer routes derive `user_id` from JWT claims instead of request form fields. |
| Bulk enrollment backend | IMPLEMENTED | Existing `/api/v1/enroll` accepts multiple files; session enrollment still receives samples one at a time. |
| Bulk enrollment UI | ROADMAP (not implemented) | The Streamlit page remains session/sample based. A true 5-file drop zone is still pending. |
| pgvector storage | IMPLEMENTED | The base schema uses pgvector for legacy voiceprints. Layer 2 encrypted anchors use encrypted blob storage plus FAISS update hooks. |
| FAISS search | IMPLEMENTED (local index) | FAISS manager exists and is updated during enrollment; pgvector remains the database-backed persistence path. |
| Kafka publishing | IMPLEMENTED (publisher only) | Events can be published when enabled; no in-repo consumer is implemented. |
| HSM provider | IMPLEMENTED (mocked provider) | Mock HSM/key-manager interface exists. AWS CloudHSM/PKCS#11 providers are stubs and require production integration. |
| INTERNAL_SERVICE_TOKEN | ROADMAP (hardening pending) | Still a static shared secret setting. Production should move to mTLS or OAuth2 client credentials. |
| Replay guard for audio samples | PARTIAL | JWT JTI replay guard exists; audio-content hash replay rejection is not yet implemented. |
| Emergency contact flow | IMPLEMENTED | Registered contact + OTP creates scoped emergency access. |
| Branch override emergency ladder | ROADMAP (not implemented) | Mock CBS branch override flow is not yet built. |
| Legal guardian workflow | ROADMAP (not implemented) | Manual dual-approval legal guardian flow is not yet built. |
| Emergency contact cooldown | ROADMAP (not implemented) | Config is present; registration/activation enforcement is pending. |
| Enrollment channel quality threshold adjustment | ROADMAP (not implemented) | Anchor records have `enrolled_via`; per-channel scoring adjustment is pending. |

## Banking Flow

1. Customer logs in with `account_number` and 6-digit MPIN.
2. The API verifies the bcrypt MPIN hash and issues a JWT with `sub`, `account_number`, `full_name`, `role`, `exp`, and `jti`.
3. Voice, consent, and emergency endpoints resolve the internal UUID from `claims["sub"]`.
4. Customers never type or see the UUID in the Streamlit journey.

## Verification

Current verified test command:

```powershell
python -m pytest tests/ -v
```

Result: 61 passed.
