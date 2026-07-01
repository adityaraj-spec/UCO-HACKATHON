# PhaseGuard Layer 2 — Architecture Explained Simply 🎙️🔐

> **"Your voice IS your password"** — but stored in a way that can never be stolen or reversed.

---

## 🧩 The Big Picture

PhaseGuard Layer 2 is a **voice-based login system** for banking. Instead of typing a PIN or OTP, you simply speak — and the system verifies it's really you.

---

## 📋 How It Works — Step by Step

### Step 1 — Sign Up (Enrollment)

- You speak **5 different sentences** into the app
- The system cleans each audio (removes noise, silence, bad recordings)
- It extracts your **voiceprint** — ~192 numbers that uniquely describe your voice, like a fingerprint
- Before saving, it **scrambles the voiceprint** using your personal secret key (called **BioHash**) — so even if someone hacks the database, they can't reconstruct your actual voice
- Everything is then **AES-256 encrypted** and saved securely

---

### Step 2 — Login (Authentication) — 8 Security Layers

When you try to login by speaking:

| Step | What Happens | Why It Matters |
|---|---|---|
| **1. Challenge** | System gives you a random phrase to say (e.g. _"My transfer is urgent"_) | So a recording of your voice from yesterday doesn't work |
| **2. Replay Check** | Token is valid only once; second use is blocked | Prevents someone replaying a captured audio |
| **3. Fraud Scan** | Checks your IP address, device, and location | Catches unusual access from unknown devices |
| **4. Audio Quality** | Checks the recording has enough voice signal | Rejects background noise or silent files |
| **5. Liveness Check** | AI detects if it's a real human voice or a deepfake/recording | Prevents voice cloning attacks |
| **6. Voiceprint Match** | Compares your voice to encrypted stored templates | The actual identity check |
| **7. Decision** | **PASS** → logged in &nbsp;·&nbsp; **STEP-UP** → OTP sent &nbsp;·&nbsp; **FAIL** → blocked | Adaptive response based on confidence |
| **8. Audit Trail** | Every action logged in a tamper-proof cryptographic chain | Like a blockchain receipt — cannot be altered |

---

### Step 3 — Smart Threshold (Adaptive)

The system **adapts to your voice over time**:

- 🤒 Have a cold? Threshold relaxed slightly so you're not locked out
- 🔒 3 failed attempts? Threshold tightens automatically
- 🌐 Suspicious IP? Security hardens even further

---

### Step 4 — Emergency Access

If you can't speak (injured, hospitalized), a **pre-registered trusted contact** (e.g. a family member) can request **72-hour limited access** on your behalf — verified by OTP.

---

### Step 5 — Privacy & Compliance

- Your raw voice is **never stored** — only a scrambled, encrypted mathematical representation
- You can **withdraw consent** anytime → all voiceprints are immediately purged from all systems
- Every action is logged for **RBI audit** readiness (India's central banking regulator)
- Compliant with **DPDP Act** (India's data protection law)

---

## 🔌 External Systems

```
Your Phone
    ↓
PhaseGuard API
    ├─→ Redis        — Fast cache for sessions & challenge tokens
    ├─→ FAISS        — Superfast vector search for voice matching
    ├─→ Kafka        — Real-time security alerts to bank SIEM
    ├─→ HSM          — Secure hardware key storage for encryption
    └─→ PostgreSQL   — All data, encrypted at rest
```

---

## 🎯 One-Line Summary

> **PhaseGuard Layer 2 is a voice login system that uses AI to recognise your voice, scrambles your voiceprint so it can't be stolen, and wraps it in 8 layers of security checks — fully compliant with Indian banking (RBI) and privacy (DPDP) regulations.**
