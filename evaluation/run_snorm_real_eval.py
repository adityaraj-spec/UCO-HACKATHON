from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings
from app.ml.ecapa_service import ECAPAService
from app.services.biohash_service import BioHashService
from app.services.snorm_service import MIN_REAL_SNORM_COHORT_SIZE, SNormService
from evaluation.metrics import wilson_interval
from evaluation.run_kathbath_part_a import (
    CONDITIONS,
    REPORTS_ROOT,
    SIM_ROOT,
    TRIAL_PAIRS,
    cached_embedding,
    collect_enrollment_paths,
)


TRIAL_SCORES = REPORTS_ROOT / "snorm_real_trial_scores.csv"
FINAL_CSV = REPORTS_ROOT / "snorm_real_eval_final.csv"
RECOMMENDATION_JSON = REPORTS_ROOT / "final_model_recommendation.json"
BLOCKED_JSON = REPORTS_ROOT / "snorm_real_eval_blocked.json"


class CohortSession:
    """Tiny async session facade so SNormService.compute_snorm_score is used unchanged."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings

    async def execute(self, _statement):
        result = MagicMock()
        result.scalars.return_value.all.return_value = self.embeddings
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-cohort S-Norm evaluation on Kathbath trials.")
    parser.add_argument("--trial-pairs", type=Path, default=TRIAL_PAIRS)
    parser.add_argument("--sim-root", type=Path, default=SIM_ROOT)
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=Path("evaluation/cache/snorm_real_eval"))
    parser.add_argument("--baseline-model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--baseline-save-dir", default="./pretrained_models/ecapa")
    parser.add_argument("--trained-model-source", default="evaluation/model_checkpoints/trained_ecapa")
    parser.add_argument("--trained-save-dir", default="evaluation/model_checkpoints/trained_ecapa")
    parser.add_argument("--force-rescore", action="store_true")
    return parser.parse_args()


def main() -> int:
    from app.utils.windows_symlink_patch import apply_windows_symlink_fallback

    apply_windows_symlink_fallback()
    args = parse_args()
    args.reports_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    try:
        trials = read_trials(args.trial_pairs)
        if not trials:
            raise RuntimeError(f"No trials found at {args.trial_pairs}")

        models = {
            "baseline": (args.baseline_model_source, args.baseline_save_dir),
            "trained": (args.trained_model_source, args.trained_save_dir),
        }
        scores = asyncio.run(score_models(models, trials, args))
        write_trial_scores(scores, args.reports_root / TRIAL_SCORES.name)
        summary = summarise(scores)
        write_summary(summary, args.reports_root / FINAL_CSV.name)
        write_recommendation(summary, args.reports_root / RECOMMENDATION_JSON.name)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "blocked",
            "reason": exc.__class__.__name__,
            "detail": str(exc),
            "required_artifacts_not_fabricated": True,
            "config_modified": False,
        }
        (args.reports_root / BLOCKED_JSON.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"S-Norm real evaluation blocked: {exc}")
        return 1

    return 0


async def score_models(
    models: dict[str, tuple[str, str]],
    trials: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, str | float]]:
    settings = get_settings()
    biohash_service = BioHashService()
    snorm_service = SNormService(biohash_service)
    enrollment_paths = collect_enrollment_paths(args.sim_root)
    if len(enrollment_paths) < MIN_REAL_SNORM_COHORT_SIZE + 2:
        raise RuntimeError(
            f"Need at least {MIN_REAL_SNORM_COHORT_SIZE + 2} real held-out speakers; "
            f"found {len(enrollment_paths)}."
        )

    all_audio_paths = sorted(
        {path for paths in enrollment_paths.values() for path in paths}
        | {Path(row["test_path"]) for row in trials}
    )
    rows: list[dict[str, str | float]] = []

    for model_name, (model_source, save_dir) in models.items():
        service = ECAPAService(model_source=model_source, save_dir=save_dir)
        model_cache = args.cache_root / model_name
        model_cache.mkdir(parents=True, exist_ok=True)
        file_embeddings = {
            path: cached_embedding(service, path, model_cache, args.force_rescore)
            for path in all_audio_paths
        }
        enrollments = {
            speaker: np.mean([file_embeddings[path] for path in paths], axis=0).astype(np.float32)
            for speaker, paths in enrollment_paths.items()
        }
        enrollment_biohashes = {
            speaker: biohash_service.compute_biohash(embedding)
            for speaker, embedding in enrollments.items()
        }

        started = time.time()
        for index, trial in enumerate(trials, start=1):
            live_embedding = file_embeddings[Path(trial["test_path"])]
            live_biohash = biohash_service.compute_biohash(live_embedding)
            enrolled_biohash = enrollment_biohashes[trial["claimed_speaker"]]
            raw_similarity = biohash_service.biohash_similarity(live_biohash, enrolled_biohash)
            cohort_embeddings = [
                embedding.astype(float).tolist()
                for speaker, embedding in enrollments.items()
                if speaker not in {trial["claimed_speaker"], trial["test_speaker"]}
            ][: settings.IMPOSTOR_COHORT_SIZE]
            if len(cohort_embeddings) < MIN_REAL_SNORM_COHORT_SIZE:
                raise RuntimeError(
                    f"Real S-Norm cohort too small for trial {trial['trial_id']}: "
                    f"{len(cohort_embeddings)} real impostors."
                )
            snorm_score, mu_imp, sigma_imp = await snorm_service.compute_snorm_score(
                CohortSession(cohort_embeddings),
                raw_similarity,
                live_biohash,
            )
            decision, verified = snorm_service.evaluate_decision(snorm_score)
            out = dict(trial)
            out.update(
                {
                    "model": model_name,
                    "raw_similarity": raw_similarity,
                    "snorm_score": snorm_score,
                    "mu_imp": mu_imp,
                    "sigma_imp": sigma_imp,
                    "decision": decision,
                    "verified": str(verified).lower(),
                    "real_impostor_cohort_size": len(cohort_embeddings),
                }
            )
            rows.append(out)
            if index % 500 == 0:
                elapsed = max(time.time() - started, 0.001)
                print(f"{model_name}: S-Norm scored {index}/{len(trials)} trials at {index / elapsed:.1f}/sec")

    return rows


def summarise(scores: list[dict[str, str | float]]) -> list[dict[str, str | int | float]]:
    rows = []
    for model in sorted({str(row["model"]) for row in scores}):
        for condition in [item.name for item in CONDITIONS]:
            subset = [row for row in scores if row["model"] == model and row["condition"] == condition]
            for trial_type in ("genuine", "impostor"):
                typed = [row for row in subset if row["trial_type"] == trial_type]
                if not typed:
                    continue
                total = len(typed)
                verified = sum(row["decision"] == "verified" for row in typed)
                step_up = sum(row["decision"] == "step_up" for row in typed)
                mismatch = sum(row["decision"] == "mismatch" for row in typed)
                same_cohort = [
                    row for row in typed if str(row["is_same_cohort"]).lower() == "true"
                ]
                if trial_type == "impostor":
                    same_false_accepts = sum(row["decision"] == "verified" for row in same_cohort)
                    same_ci_low, same_ci_high = wilson_interval(same_false_accepts, len(same_cohort))
                else:
                    same_false_accepts = 0
                    same_ci_low, same_ci_high = 0.0, 0.0
                ci_low, ci_high = wilson_interval(verified, total)
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "trial_type": trial_type,
                        "total_trials": total,
                        "verified_rate_percent": 100.0 * verified / total,
                        "verified_ci95_low_percent": 100.0 * ci_low,
                        "verified_ci95_high_percent": 100.0 * ci_high,
                        "step_up_rate_percent": 100.0 * step_up / total,
                        "mismatch_rate_percent": 100.0 * mismatch / total,
                        "same_cohort_impostor_trials": len(same_cohort) if trial_type == "impostor" else 0,
                        "same_cohort_false_accepts": same_false_accepts,
                        "same_cohort_far_percent": (
                            100.0 * same_false_accepts / len(same_cohort)
                            if trial_type == "impostor" and same_cohort
                            else 0.0
                        ),
                        "same_cohort_far_ci95_low_percent": 100.0 * same_ci_low,
                        "same_cohort_far_ci95_high_percent": 100.0 * same_ci_high,
                    }
                )
    return rows


def write_recommendation(summary: list[dict[str, str | int | float]], path: Path) -> None:
    genuine = [row for row in summary if row["trial_type"] == "genuine"]
    impostor = [row for row in summary if row["trial_type"] == "impostor"]

    def avg(rows, model, key):
        values = [float(row[key]) for row in rows if row["model"] == model]
        return sum(values) / len(values) if values else None

    baseline_genuine = avg(genuine, "baseline", "verified_rate_percent")
    trained_genuine = avg(genuine, "trained", "verified_rate_percent")
    baseline_far = avg(impostor, "baseline", "same_cohort_far_percent")
    trained_far = avg(impostor, "trained", "same_cohort_far_percent")

    recommendation = "inconclusive"
    rationale = "Both model families must be compared on genuine acceptance and same-cohort false accepts."
    if None not in {baseline_genuine, trained_genuine, baseline_far, trained_far}:
        if trained_far <= baseline_far and trained_genuine >= baseline_genuine:
            recommendation = "promote_fine_tuned_model"
            rationale = "Fine-tuned model is no worse on same-cohort false accepts and no worse on genuine verification."
        elif baseline_far <= trained_far and baseline_genuine >= trained_genuine:
            recommendation = "keep_baseline"
            rationale = "Baseline is no worse on same-cohort false accepts and no worse on genuine verification."

    payload = {
        "recommendation": recommendation,
        "rationale": rationale,
        "evidence": {
            "baseline_avg_genuine_verified_rate_percent": baseline_genuine,
            "trained_avg_genuine_verified_rate_percent": trained_genuine,
            "baseline_avg_same_cohort_far_percent": baseline_far,
            "trained_avg_same_cohort_far_percent": trained_far,
        },
        "source_artifacts": {
            "summary_csv": str(FINAL_CSV),
            "trial_scores_csv": str(TRIAL_SCORES),
        },
        "config_modified": False,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_trials(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_trial_scores(rows: list[dict[str, str | float]], path: Path) -> None:
    fieldnames = [
        "model",
        "trial_id",
        "condition",
        "trial_type",
        "claimed_speaker",
        "test_speaker",
        "test_path",
        "claimed_cohort",
        "test_cohort",
        "is_same_cohort",
        "raw_similarity",
        "snorm_score",
        "mu_imp",
        "sigma_imp",
        "decision",
        "verified",
        "real_impostor_cohort_size",
    ]
    write_csv(path, rows, fieldnames)


def write_summary(rows: list[dict[str, str | int | float]], path: Path) -> None:
    write_csv(path, rows, list(rows[0].keys()) if rows else [])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
