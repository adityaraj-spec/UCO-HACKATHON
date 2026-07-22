# 🚀 PhaseGuard Production Migration Guide

This guide describes how to transition PhaseGuard Layer 2 from fixed-threshold verification to the zero-downtime, biohashed, adaptive score-normalised production architecture.

---

## 1. Database Migration & BioHash Backfill

Run Alembic database migration to create new compliance, audit, and impostor cohort tables:

```bash
alembic upgrade head
```

Run the data migration script to convert all existing raw vector voiceprints into BioHashes:

```bash
python migrate_voiceprints_to_biohashes.py
```

---

## 2. FAISS Index Training & Initialization

Train the offline `IndexIVFPQ + HNSW` quantizer:

```bash
python train_faiss_index.py
```

---

## 3. Environment Variable Configuration

Ensure the following production settings are configured in `.env` or system environment:

```env
# Score Normalisation (s-norm)
SNORM_PASS_THRESHOLD=3.0
SNORM_STEP_UP_THRESHOLD=1.5
SNORM_HARD_FAIL_THRESHOLD=0.0
IMPOSTOR_COHORT_SIZE=200

# BioHash Settings
BIOHASH_SEED_KEY=YourProductionSecretSeedKey2026
BIOHASH_DIM=256

# FAISS IVF-PQ Settings
FAISS_INDEX_PATH=./data/faiss_ivfpq.index
FAISS_NPROBE=32
FAISS_M=16
FAISS_NBITS=8

# Audio Preprocessing, Diarisation & Quality Gating
ENABLE_AUDIO_QUALITY_CHECK=True
MIN_SPEECH_DURATION_SEC=2.0
EMBEDDING_CONFIDENCE_THRESHOLD=0.5
TARGET_NORM_SCALE=10.0

# Liveness & Replay Protection
ENABLE_CHALLENGE_ASR=True
ENABLE_PERCEPTUAL_REPLAY_CHECK=True
CHALLENGE_TTL_SEC=60
AUDIO_HASH_TTL_SEC=86400

# Observability
ENABLE_OPENTELEMETRY=True
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## 4. Verification & Health Monitoring

Verify system readiness using the Kubernetes health probe endpoint:

```bash
curl http://localhost:8000/healthz
```

Expected response:
```json
{
  "status": "healthy",
  "redis_connected": true,
  "faiss_indexed": true,
  "faiss_total": 1000
}
```

View Prometheus metrics:
```bash
curl http://localhost:8000/metrics
```
