# Architecture Decisions

## pgvector and FAISS

PhaseGuard uses PostgreSQL as the durable system of record. The original `voiceprints` table uses pgvector, while the Layer 2 anchor vault stores encrypted BioHash templates and updates a local FAISS index for fast nearest-neighbour lookup.

Decision: PostgreSQL remains the primary persistence backend. FAISS is treated as an acceleration layer/local index, not the compliance record of truth.

Why: pgvector-backed PostgreSQL is easier to back up, audit, migrate, and operate in a banking environment. FAISS is valuable for low-latency search and high-scale experiments, but it should be rebuilt from trusted stored data rather than treated as the canonical vault.

## Kafka

Kafka support is publisher-only in this repository. `app/fraud/kafka_publisher.py` can emit events when `KAFKA_ENABLED=True`, but there is no in-repo consumer.

Decision: document Kafka as "publisher implemented; downstream consumer is roadmap or owned by the bank/SIEM team."

## HSM

The current HSM path is a mocked provider. The key-manager interface supports swapping to `aws_cloudhsm` or `pkcs11`, but the production providers are stubs and require real vendor configuration and implementation.

Decision: label HSM as `IMPLEMENTED (mocked provider)` until a real HSM adapter is completed and tested.

## Internal Service Token

V1 uses `INTERNAL_SERVICE_TOKEN` as a static shared secret for Layer 1 to Layer 2 service calls.

Decision: keep this only as a development/pilot integration mechanism. The production upgrade path is mTLS or OAuth2 client credentials between Layer 1 and Layer 2 services, with token rotation and issuer/audience validation.

## Replay Guard

JWT JTI replay protection exists, and `/auth/login` now supports nonce/timestamp replay rejection. Audio-content replay detection is still pending.

Decision: do not claim full anti-replay coverage until authentication audio hashes are cached and duplicate clips are rejected within a short TTL window.

## Branch Code

`branch_code` is the customer's registered home branch captured during KYC. It does not imply a physical branch visit for voice enrollment. That is separate from any future high-friction branch emergency override workflow.
