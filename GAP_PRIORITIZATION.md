| Bucket | Feature | Current Status | Fix Decision |
|---|---|---|---|
| Security-critical | Consent enforced before all enrollment paths | ❌ MISSING | Enforce active consent inside EnrollmentService.enroll and add a regression test for /enroll-style service calls. |
| Security-critical | KYC gating for backend enrollment | ⚠️ PARTIAL | Block enrollment for REJECTED/PENDING users in EnrollmentService, matching Streamlit policy. |
| Security-critical | INTERNAL_SERVICE_TOKEN on /voice/authenticate/direct | ❌ MISSING | Add an internal-service-token dependency and require it on direct voice auth. |
| Security-critical | Challenge JTI replay registration | ⚠️ PARTIAL | Register challenge JWT JTI during /voice/challenge issuance so AuthService can consume it once. |
| Security-critical | MPIN auth rate limiting | ⚠️ PARTIAL | Apply SlowAPI limits to register/login/change endpoints. |
| Security-critical | Emergency OTP registration runtime bug | ⚠️ PARTIAL | Import MockOTPProvider or avoid provider-specific type dependency. |
| Security-critical | Emergency max contacts setting | ⚠️ PARTIAL | Use settings.EMERGENCY_MAX_CONTACTS instead of hardcoded 2. |
| Security-critical | Audit vault call-site coverage for active lifecycle events | ⚠️ PARTIAL | Add vault logs for consent grant/withdraw, challenge issued, auth attempt, contact register, and fraud flag paths. |
| Core-functionality | Bulk enrollment UI | ⚠️ PARTIAL | Add multi-file mode to Streamlit enrollment while keeping session-based flow available. |
| Core-functionality | Legacy raw user_id client helpers/API docs | ⚠️ PARTIAL | Remove user_id payload from Streamlit legacy helpers and align descriptions with JWT-derived user identity. |
| Core-functionality | Branch override endpoint | ❌ MISSING | Descope as roadmap unless required for current demo; legal-guardian workflow is present but separate. |
| Core-functionality | ENROLLMENT_CHANNEL config | ❌ MISSING | Add config default and thread enrollment channel into metadata/threshold where practical. |
| Nice-to-have/roadmap | Real HSM integration | ⚠️ PARTIAL | Keep KeyManager abstraction and document AWS/PKCS#11 integration requirements. |
| Nice-to-have/roadmap | True refresh-token lifecycle | ⚠️ PARTIAL | Document as roadmap; current /auth/refresh is an access-token renewal helper. |
| Nice-to-have/roadmap | FAISS durable ID map and startup rehydration | ⚠️ PARTIAL | Document persistence/rehydration design; current in-memory map works only for live process. |
| Nice-to-have/roadmap | Kafka consumer auto-start and production SIEM callback | ⚠️ PARTIAL | Document deployment choice; publisher path and consumer class exist. |
| Nice-to-have/roadmap | noisereduce package specifically | ⚠️ PARTIAL | Keep existing Wiener-filter suppressor unless product explicitly requires noisereduce. |
| Nice-to-have/roadmap | BRANCH_OVERRIDE and LEGAL_GUARDIAN enum tiers | ⚠️ PARTIAL | Legal guardian is modeled as workflow endpoints; branch override remains roadmap. |
