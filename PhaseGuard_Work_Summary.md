# PhaseGuard Layer 2 — What Was Built & Updated
> A plain-English summary of all changes made to the PhaseGuard project.  
> You can share this document with anyone to explain the work done.

---

## 🧠 What is PhaseGuard?

PhaseGuard is a **voice-based authentication system** built for Indian banking, designed to comply with **RBI regulations** and **India's DPDP (Digital Personal Data Protection) Act**.

Instead of a password or PIN, the user **speaks a phrase** and the system matches their voice to verify identity — like a fingerprint, but for your voice.

It has **two layers**:
- **Layer 1** — A machine learning model (ECAPA-TDNN) that creates a "voice fingerprint" from an audio recording.
- **Layer 2** — The full production system built on top of Layer 1: enrollment, authentication, fraud detection, compliance, and security.

---

## ✅ What Was Built (Layer 2 — Session 1)

### 1. 🎙️ Voice Enrollment Pipeline
**What it does:** Registers a new user's voice into the system.

- The user must record **5 voice samples**.
- Each sample is checked for quality (no noise, long enough, voice is actually present).
- A "voice fingerprint" (embedding) is created and encrypted using **AES-256-GCM** (military-grade encryption).
- Two copies are stored:
  - **Anchor** — the original, never changes.
  - **Rolling Pool** — up to 10 recent samples that update over time to handle natural voice changes.
- Raw audio files are **immediately deleted** after processing (privacy compliance).

---

### 2. 🔐 Voice Authentication Pipeline
**What it does:** Verifies if the person speaking is really who they say they are.

Step-by-step flow:
1. Server sends a **random challenge phrase** (e.g. "River mountain cloud seven") — so you can't replay an old recording.
2. User speaks the phrase.
3. Audio is cleaned up (noise removed, silence cut).
4. **Liveness check** — is this a real human or a recording/AI voice?
5. Voice fingerprint is extracted.
6. Compared against stored templates using weighted scoring:
   - **40% weight** on anchor (original voice).
   - **60% weight** on recent rolling pool (adapts to voice changes).
7. Decision: **PASS / STEP_UP / FAIL**.

---

### 3. 📊 Adaptive Threshold Engine
**What it does:** Adjusts the strictness of voice matching per-user.

- If you're slightly sick and your voice changed, the system **widens the threshold** instead of blocking you.
- If someone has been failing repeatedly, the threshold **tightens** as a fraud signal.
- Each user has their own personalized threshold stored in the database.

---

### 4. 🛡️ Anti-Fraud & Risk Engine
**What it does:** Checks context around each login attempt.

- Looks at **IP address reputation**, **device fingerprint**, and **location velocity** (e.g., login from Delhi, then Mumbai 2 minutes later = suspicious).
- Produces a **risk score** for each attempt.
- If risk is high, triggers a **step-up challenge** (additional verification).
- High-risk events are published to **Kafka** for SIEM (security monitoring) systems.

---

### 5. 🔒 Security Middleware
**What it does:** Protects every API endpoint.

| Layer | What it does |
|---|---|
| **JWT Auth** | Every request needs a valid signed token |
| **RBAC** | Different roles (Customer / Agent / Admin) get different access |
| **Rate Limiter** | Max 3 enrollment attempts/hour, 30 auth attempts/minute |
| **Replay Guard** | Each token can only be used once (stored in Redis) |

---

### 6. ✅ Consent Management (DPDP Compliance)
**What it does:** Ensures users explicitly agree before their biometric data is stored.

- User must **grant consent** before any voice enrollment begins.
- Consent can be **withdrawn** at any time — this auto-deletes all stored voice templates.
- Full consent history is logged.

---

### 7. 🚨 Emergency Access Framework
**What it does:** Lets a trusted contact temporarily access an account if the user can't authenticate (accident, illness, etc.).

- User registers an **emergency trustee** contact (OTP-verified).
- In an emergency, trustee can activate a **72-hour override session**.
- All access is logged in the audit trail.

---

### 8. 📜 Audit Vault (Immutable Logs)
**What it does:** Records every action in a tamper-proof log.

- Every enrollment, authentication, consent change, and emergency access is recorded.
- Logs are chained with **SHA-256 hashes** — if anyone modifies a record, the chain breaks.
- Admin-only access to retrieve full audit trails per user.

---

### 9. 🗄️ Database Schema (12 New Tables)
New tables created for Layer 2:

| Table | Purpose |
|---|---|
| `enrollments` | Tracks enrollment sessions |
| `anchor_embeddings` | Stores the original voice template |
| `rolling_embeddings` | Stores recent voice samples |
| `auth_sessions` | Tracks active authentication sessions |
| `auth_history` | All past authentication results |
| `consents` | Consent records per user |
| `audit_logs` | Immutable chained audit trail |
| `challenge_phrases` | Issued challenge tokens |
| `emergency_contacts` | Registered trustees |
| `emergency_access_log` | Emergency session activity |
| `user_thresholds` | Per-user adaptive thresholds |
| `voice_metadata` | Audio quality metadata |

---

### 10. 🌐 API Endpoints Built

| Endpoint | What it does |
|---|---|
| `POST /consent/grant` | User opts in to biometric storage |
| `POST /consent/withdraw` | User deletes all their biometric data |
| `POST /voice/enroll/session` | Start a new enrollment |
| `POST /voice/enroll/sample` | Upload one voice sample |
| `POST /voice/enroll/complete` | Finish enrollment |
| `GET /voice/challenge` | Get a challenge phrase to speak |
| `POST /voice/authenticate` | Authenticate by voice |
| `POST /emergency/contact` | Register an emergency trustee |
| `POST /emergency/activate` | Start emergency override |
| `GET /audit/user/{id}` | View audit log (Admin only) |
| `GET /health/layer2` | System health check |

---

### 11. 🧪 Test Suite
Tests written to verify each component works correctly:
- Audio quality checks (SNR, VAD)
- BioHash transformation (cancelable biometrics)
- AES-256 encryption round-trips
- Consent workflows
- Threshold engine logic
- Emergency contact and OTP flows
- Fraud engine scoring

---

## 🔧 What Was Fixed (Session 2 — Today)

### Streamlit Frontend Deprecation Warnings
**Problem:** The console was printing:
```
Please replace `use_container_width` with `width`.
`use_container_width` will be removed after 2025-12-31.
```

**What was changed:** Updated 3 frontend pages to use the new Streamlit API:

| File | Occurrences Fixed |
|---|---|
| `pages/4_Risk_Dashboard.py` | 4 charts updated |
| `pages/3_History.py` | 2 data tables updated |
| `pages/2_Verification.py` | 2 charts updated |

**Before:**
```python
st.plotly_chart(fig, use_container_width=True)
```
**After:**
```python
st.plotly_chart(fig, width='stretch')
```

This is purely a **code hygiene fix** — no visual change to the dashboard, just silences the deprecation warning.

---

## 🏗️ System Architecture (Simple View)

```
User speaks phrase
       ↓
API receives audio
       ↓
Fraud check (IP, device, location)
       ↓
Audio cleaning (noise removal, VAD)
       ↓
Liveness check (real human?)
       ↓
Voice fingerprint extracted (ECAPA-TDNN model)
       ↓
BioHash transformation (privacy layer)
       ↓
Compared to stored templates (40% anchor + 60% rolling)
       ↓
Adaptive threshold applied
       ↓
Decision: PASS / STEP_UP / FAIL
       ↓
Audit log written → Kafka event published
```

---

## 🖥️ How to Run

```bash
# 1. Start the backend API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Start the Streamlit dashboard (in /streamlit_app folder)
streamlit run app.py

# 3. Open browser
# API docs:  http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| Voice Model | ECAPA-TDNN (SpeechBrain) |
| Database | PostgreSQL (via SQLAlchemy) |
| Cache / Sessions | Redis |
| Event Streaming | Apache Kafka |
| Vector Search | FAISS |
| Encryption | AES-256-GCM |
| Frontend Dashboard | Streamlit |
| Containerization | Docker + Docker Compose |

---

*Document generated: 2026-07-01 | Project: PhaseGuard Layer 2 | Status: Production-ready*
