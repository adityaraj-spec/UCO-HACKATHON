# PhaseGuard Final Consolidated Report

Generated on 2026-07-22 from the current `UCO-HACKATHON` workspace.

## Part C - Real S-Norm Evaluation

Artifacts:

- `evaluation/run_snorm_real_eval.py`
- `evaluation/reports/snorm_real_trial_scores.csv`
- `evaluation/reports/snorm_real_eval_final.csv`
- `evaluation/reports/final_model_recommendation.json`

The evaluation used the existing Kathbath held-out trial pairs and telephone conditions. It computed file embeddings for both baseline VoxCeleb ECAPA and the local 4-epoch trained checkpoint, converted embeddings through `BioHashService`, used real held-out speaker enrollment embeddings as S-Norm impostor cohorts, and called `SNormService.compute_snorm_score()` plus `evaluate_decision()`.

Synthetic S-Norm fallback was not used. Trial rows show real cohort sizes of 38 or 39 impostor speakers.

Summary from `snorm_real_eval_final.csv`:

| Model | Condition | Genuine verified % | Impostor verified % | Same-cohort FAR % |
| --- | --- | ---: | ---: | ---: |
| baseline | clean | 90.0 | 1.0 | 1.2539 |
| baseline | telephone_mild | 40.0 | 0.75 | 0.9231 |
| baseline | telephone_moderate | 42.5 | 0.5 | 0.5988 |
| baseline | telephone_severe | 35.0 | 1.25 | 1.5198 |
| trained | clean | 92.5 | 1.0 | 1.2539 |
| trained | telephone_mild | 40.0 | 0.5 | 0.6154 |
| trained | telephone_moderate | 32.5 | 0.75 | 0.8982 |
| trained | telephone_severe | 25.0 | 1.0 | 1.2158 |

Recommendation from `final_model_recommendation.json`: `inconclusive`.

Evidence:

- Baseline average genuine verified rate: 51.875%.
- Trained average genuine verified rate: 47.5%.
- Baseline average same-cohort FAR: 1.0739%.
- Trained average same-cohort FAR: 0.9958%.

Interpretation: the trained checkpoint slightly reduced average same-cohort FAR but also reduced genuine acceptance, especially under moderate/severe telephone conditions. Do not promote the fine-tuned checkpoint without further training/evaluation.

## Part D - Enrollment Security

Implemented:

- `app/services/layer1_liveness_service.py` runs the existing Layer 1 model for enrollment.
- `EnrollmentService.enroll()` now saves each upload, loads waveform, applies enrollment audio quality gating, runs Layer 1, and rejects if AI probability exceeds `settings.LAYER1_FRAUD_THRESHOLD`.
- `app/utils/audio_quality.py` now includes enrollment duration and estimated SNR gates.
- `EnrollmentRejectedError` is handled by the enrollment endpoint as HTTP 422.
- `tests/test_enrollment_service.py` includes a synthetic/deepfake rejection test.

Operational note: enrollment is fail-closed if `LAYER1_MODEL_PATH` is absent. In this workspace, `models/layer1_mobilenet.pth` is not present, so real enrollment needs that model artifact before production use.

Diarisation threshold review:

- Artifact: `evaluation/reports/diarisation_threshold_review.json`.
- Finding: no evidence-backed change to the hardcoded `0.65` diarisation clustering threshold. The corrected S-Norm evaluation uses file-level embeddings and does not emit diarisation cluster labels.
- Required next artifact: a multi-speaker diarisation validation set with frame/window labels and telephone-condition stratification.

## Part E - Encryption At Rest

Implemented:

- `app/services/voiceprint_crypto.py` provides AES-256-GCM encryption for embeddings and BioHash strings.
- `VoiceprintRepository.upsert()` encrypts the embedding and BioHash before persistence.
- `Voiceprint` keeps legacy raw columns nullable and attaches decrypted plaintext only in-process.
- Migration: `alembic/versions/20260722_0100_voiceprint_encryption.py`.
- Tests: `tests/test_voiceprint_crypto.py`.

Required production secret:

- `VOICEPRINT_ENCRYPTION_KEY_B64`: base64-encoded 32-byte AES key.
- Optional `VOICEPRINT_ENCRYPTION_KEY_ID`.

No `.env` or config file was modified. FAISS remains an ephemeral in-memory cache fed from plaintext in process after decryption/rebuild.

## Part F - FAISS Scaling

Implemented:

- `FAISSService` now supports configurable `flat`, `hnsw`, and `ivfpq` index creation via constructor or `FAISS_INDEX_TYPE`.
- `benchmarks/faiss_scaling_benchmark.py` measures local p50/p95/p99 and recall@k.
- `evaluation/reports/faiss_multi_instance_design.md` documents a multi-instance snapshot/reload strategy.

Measured local synthetic-vector benchmark from `evaluation/reports/faiss_scaling_benchmark.csv`:

| Index | Vectors | p50 ms | p95 ms | p99 ms | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | 1,000 | 0.0166 | 0.0342 | 0.0754 | 1.000 |
| hnsw | 1,000 | 0.0414 | 0.0553 | 0.1058 | 1.000 |
| ivfpq | 1,000 | 0.0434 | 0.0465 | 0.0642 | 0.419 |
| flat | 10,000 | 0.1650 | 0.2339 | 0.5799 | 1.000 |
| hnsw | 10,000 | 0.1313 | 0.1605 | 0.1907 | 0.934 |
| ivfpq | 10,000 | 0.0803 | 0.0912 | 0.1246 | 0.199 |
| flat | 100,000 | 2.7402 | 3.2101 | 4.3108 | 1.000 |
| hnsw | 100,000 | 0.4036 | 0.4743 | 0.5335 | 0.524 |
| ivfpq | 100,000 | 0.2093 | 0.2212 | 0.2349 | 0.133 |

Interpretation: HNSW gives a useful latency reduction at 100k but needs recall tuning (`efSearch`, graph parameters). The current IVFPQ settings are fast but low recall and are not ready as a default identity-search index without tuning.

## Part G - Verify Latency

Artifacts:

- `benchmarks/verify_latency_benchmark.py`
- `evaluation/reports/verify_latency_benchmark.json`
- `evaluation/reports/verify_latency_benchmark.csv`

Status: blocked for real end-to-end measurement.

Reason: no reachable live API plus valid enrolled user/audio context was available in this workspace. The harness attempted one request and recorded an all-error run, then wrote a blocked artifact.

Stage-level latency breakdown is therefore not claimed. To measure it honestly, the API must be running with PostgreSQL/Redis/model artifacts, an enrolled user, and server-side timing spans around diarisation, embedding extraction, BioHash, S-Norm cohort fetch, decision, and logging.

## Test Status

Command run:

```text
pytest -q --basetemp=.pytest_tmp
```

Result:

```text
65 passed, 1 warning
```

The warning is the pre-existing pytest cache path issue on this Windows/OneDrive workspace.

## Production-Ready vs Roadmap

Genuinely improved in this pass:

- Real S-Norm evaluation artifacts exist and are grounded in held-out Kathbath trials.
- Enrollment now blocks synthetic/deepfake audio and low-quality samples before voiceprint creation.
- Voiceprint embedding and BioHash encryption at rest is implemented and test-covered.
- FAISS can now use HNSW/IVFPQ alternatives and has local benchmark numbers.

Still roadmap / not production-ready:

- Fine-tuned checkpoint promotion is not supported by the current evidence.
- Layer 1 model artifact is absent locally, so real enrollment must provision it.
- Diarisation threshold `0.65` cannot be recalibrated without diarisation-specific labeled data.
- IVFPQ recall is too low with current settings.
- End-to-end `/api/v1/verify` latency and stage breakdown remain blocked until a live enrolled API environment exists.
