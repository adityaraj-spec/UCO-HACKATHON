# Production Decision Path Confirmed

Generated on 2026-07-22 after inspecting the current checkout under `UCO-HACKATHON`.

## A.1 Live verification decision path

The live route is `POST /api/v1/verify`, mounted under `/api/v1` by `app/main.py`.
There is no path-parameter route named `/api/v1/verify/{user_id}` in the current code; the user id is accepted as a form field.

Exact call chain:

1. `app/main.py:181` mounts `api_router` with prefix `/api/v1`.
2. `app/api/v1/router.py:15` includes `verify.router`.
3. `app/api/v1/endpoints/verify.py:32-45` registers `@router.post("/verify")`.
4. `app/api/v1/endpoints/verify.py:46-58` accepts `user_id` as `Form(...)`, not as a path parameter.
5. `app/api/v1/endpoints/verify.py:60-65` constructs `VerificationService(session=db)` and returns `await service.verify(...)`.
6. `app/services/verification_service.py:66-68` constructs the production service dependencies: `get_ecapa_service()`, `BioHashService()`, and `SNormService(...)`.
7. `app/services/verification_service.py:87-89` fetches the enrolled voiceprint.
8. `app/services/verification_service.py:93-99` saves the uploaded audio, loads the waveform, and extracts the dominant/target speaker audio.
9. `app/services/verification_service.py:127-128` extracts the live ECAPA embedding and computes the live BioHash.
10. `app/services/verification_service.py:131-135` uses the stored enrolled BioHash if present, or computes it from the stored enrolled embedding if missing.
11. `app/services/verification_service.py:137` computes raw similarity with `BioHashService.biohash_similarity(...)`.
12. `app/services/biohash_service.py:46-59` implements this as normalized Hamming distance between BioHash strings, mapped to `cos(pi * HammingDistance)`.
13. `app/services/verification_service.py:140-142` calls `SNormService.compute_snorm_score(...)`.
14. `app/services/snorm_service.py:35-56` fetches up to `settings.IMPOSTOR_COHORT_SIZE` real impostor embeddings, computes their BioHash similarities, and returns `(raw_similarity - mu_imp) / sigma_imp`. If fewer than 5 real cohort embeddings exist, deterministic synthetic random unit vectors top the cohort up only to the minimum size of 5.
15. `app/services/verification_service.py:143` calls `self.snorm_service.evaluate_decision(snorm_score)`.
16. `app/services/snorm_service.py:58-68` makes the returned identity decision:
    - `snorm_score >= settings.SNORM_PASS_THRESHOLD` -> `"verified", True`
    - `snorm_score >= settings.SNORM_STEP_UP_THRESHOLD` -> `"step_up", False`
    - otherwise -> `"mismatch", False`
17. `app/core/config.py:62-64` currently defines those S-Norm thresholds as `3.0`, `1.5`, and `0.0`.
18. `app/services/verification_service.py:146-149` can downgrade a `"verified"` decision to `"step_up"` if model confidence is below `settings.EMBEDDING_CONFIDENCE_THRESHOLD`.
19. `app/services/verification_service.py:175-183` returns `decision` and `verified` from the S-Norm path, possibly after the confidence downgrade.

Conclusion: the final identity `decision` and `verified` boolean returned by the live verification endpoint are made by `SNormService.evaluate_decision()`, with an additional model-confidence downgrade path. The legacy/fallback `STEP_UP_LOWER_BOUND` and `HARD_FAIL_THRESHOLD` values are not used in this live decision path.

The risk engine still runs, but it is not the source of the returned identity `decision`/`verified` fields:

1. `app/services/verification_service.py:152` calls `compute_risk(layer1_score=layer1_score, speaker_similarity=raw_similarity)`.
2. `app/services/risk_engine.py:59-92` returns only `risk_score` and `risk_level`.
3. `app/services/risk_engine.py:73-84` uses `settings.LAYER1_FRAUD_THRESHOLD` and `settings.SIMILARITY_THRESHOLD`.
4. `app/services/verification_service.py:175-183` returns these as separate `risk_score` and `risk_level` fields.
5. `app/schemas/verification.py:29-37` confirms the response has separate identity fields (`similarity_score`, `verified`, `decision`) and risk fields (`layer1_score`, `risk_score`, `risk_level`).

Important contradiction/staleness found:

- `app/api/v1/endpoints/verify.py:39-43` now describes the BioHash plus s-norm identity decision path.
- `app/ml/ecapa_service.py:206-226` still exposes `verify_user(...)` cosine scoring for compatibility, but its docstring now notes that `VerificationService.verify(...)` does not call this method in the live endpoint path.
- `app/main.py:143-149` exposes `similarity_threshold` in `/health`, but that threshold is only used by the secondary risk engine in live verification, not by the `decision`/`verified` identity result.

Search confirmation:

- `SNormService.evaluate_decision(...)` is called from `app/services/verification_service.py:143` in the live path and directly in `tests/test_production_upgrade.py`.
- `compute_risk(...)` is called from `app/services/verification_service.py:152` and by tests.
- `STEP_UP_LOWER_BOUND` and `HARD_FAIL_THRESHOLD` are defined in `app/core/config.py:57-58`, but `rg` found no live `app/` call site using them.
- No `PASS_THRESHOLD` setting exists in `app/core/config.py`; `PASS_THRESHOLD` appears only in evaluation/report artifacts.

Dead/unused status:

- `STEP_UP_LOWER_BOUND` and `HARD_FAIL_THRESHOLD` are dead/unused for the live verification request path.
- `SIMILARITY_THRESHOLD` is not dead: it is used by `risk_engine.py` to produce `risk_level`, but it is not the final identity pass/step-up/fail decision threshold.
- `ECAPAService.verify_user(...)` is not used by the live `/api/v1/verify` endpoint; it remains a helper/API compatibility path.

## A.2 Checkpoint contradiction and epoch reconciliation

At the time of this inspection, `evaluation/model_checkpoints/trained_ecapa/` exists and contains a complete-looking SpeechBrain-style checkpoint bundle:

| File | CreationTime | LastWriteTime | Size |
| --- | --- | --- | --- |
| `embedding_model.ckpt` | 2026-07-22 02:44:54 | 2026-07-22 02:55:55 | 83,312,601 bytes |
| `classifier.ckpt` | 2026-07-22 02:44:54 | 2026-07-22 02:55:55 | 32,765 bytes |
| `hyperparams.yaml` | 2026-07-22 02:44:54 | 2026-07-22 03:03:27 | 2,055 bytes |
| `mean_var_norm_emb.ckpt` | 2026-07-22 02:48:01 | 2026-06-12 07:21:25 | 1,921 bytes |
| `label_encoder.ckpt` | 2026-07-22 02:48:02 | 2026-06-12 07:21:29 | 128,619 bytes |

Timeline reconciliation:

- A prior report saying no fine-tuned checkpoint existed could have been true only before `2026-07-22 02:44:54` in this workspace, or in a different checkout/workspace.
- In the current checkout after `2026-07-22 02:44:54`, that prior statement is false: the checkpoint files are present.
- The June `LastWriteTime` values on `mean_var_norm_emb.ckpt` and `label_encoder.ckpt` indicate those files were copied from an older/pretrained source while preserving original modification times. `evaluation/train_ecapa_svarah.py:252-256` copies those ancillary files with `shutil.copy2(...)`, which preserves metadata.
- The primary trained ECAPA artifacts are `embedding_model.ckpt` and `classifier.ckpt`; both were created at `2026-07-22 02:44:54` and last written at `2026-07-22 02:55:55`.

Epoch-count reconciliation:

- The ECAPA/Svarah training script hardcodes `epochs = 4` at `evaluation/train_ecapa_svarah.py:124`.
- The ECAPA/Svarah training summary reports `"epochs": 4` in `evaluation/reports/ecapa_training_summary.json`.
- The ECAPA/Svarah training history contains exactly epochs `1` through `4` in `evaluation/reports/ecapa_training_history.csv`.
- The separate `models/training_history.csv` contains epochs `1` through `20`, but it is not referenced by the ECAPA checkpoint path. It appears to be a Layer 1/deepfake-model training artifact under `models/`, not the Layer 2 ECAPA checkpoint. Related Layer 1 training code saves `models/layer1_mobilenet.pth` in `train_layer1.py:155-159`, and another Layer 1 script saves the same model path in `scripts/train.py:105-109`.

Conclusion: the currently present `evaluation/model_checkpoints/trained_ecapa/` checkpoint corresponds to the 4-epoch Svarah ECAPA run. The 20-epoch history is a separate `models/` artifact and is not the ECAPA checkpoint currently referenced by configuration defaults.

## A.3 Runtime ECAPA model source

The actual resolved runtime value was printed by importing the FastAPI entrypoint and ECAPA singleton from the project root:

```text
app.main.settings.ECAPA_MODEL_SOURCE=speechbrain/spkrec-ecapa-voxceleb
ECAPAService.model_source=speechbrain/spkrec-ecapa-voxceleb
ECAPAService.save_dir=./pretrained_models/ecapa
```

This matters because:

- `app/core/config.py:50-51` defaults to `evaluation/model_checkpoints/trained_ecapa`.
- `.env:21-22` overrides that default to `speechbrain/spkrec-ecapa-voxceleb` and `./pretrained_models/ecapa`.
- `app/main.py:48` resolves `settings = get_settings()`.
- `app/main.py:65-66` preloads the model with `get_ecapa_service().load_model()`.
- `app/ml/ecapa_service.py:52-53` sets `self.model_source` and `self.save_dir` from the resolved settings.
- `app/ml/ecapa_service.py:89-92` passes `source=self.model_source` and `savedir=self.save_dir` to `SpeakerRecognition.from_hparams(...)`.

Conclusion: right now, application startup uses the baseline VoxCeleb SpeechBrain model source `speechbrain/spkrec-ecapa-voxceleb`, not the fine-tuned Svarah checkpoint under `evaluation/model_checkpoints/trained_ecapa`.

## Stop point

Per the mission instructions, work stops here. Parts B-F should not proceed until A.1-A.3 are reviewed.
