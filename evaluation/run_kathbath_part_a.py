from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from app.core.config import get_settings
from app.ml.ecapa_service import ECAPAService
from evaluation.metrics import compute_eer, det_points, percentile, wilson_interval
from evaluation.telephone_simulator import CONDITIONS, simulate_file


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "evaluation"
DATA_ROOT = EVAL_ROOT / "data"
REPORTS_ROOT = EVAL_ROOT / "reports"
CACHE_ROOT = EVAL_ROOT / "cache" / "kathbath_embeddings"
RAW_ROOT = DATA_ROOT / "kathbath_raw"
SIM_ROOT = DATA_ROOT / "kathbath_telephone"
KATHBATH_MANIFEST = REPORTS_ROOT / "kathbath_selection_manifest.csv"
TRIAL_PAIRS = REPORTS_ROOT / "trial_pairs_kathbath.csv"
ALL_SCORES = REPORTS_ROOT / "all_trial_scores_kathbath.csv"


@dataclass(frozen=True)
class Utterance:
    speaker_key: str
    speaker_id: str
    lang: str
    gender: str
    region: str
    utterance_index: int
    path: Path
    source_fname: str

    @property
    def cohort_key(self) -> str:
        return "|".join([self.gender or "unknown", self.lang or "unknown", self.region or "unknown"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PhaseGuard Part A evaluation on a Kathbath speaker slice."
    )
    parser.add_argument("--speaker-count", type=int, default=40)
    parser.add_argument("--min-utterances", type=int, default=6)
    parser.add_argument("--impostors-per-test", type=int, default=10)
    parser.add_argument("--same-cohort-impostors-per-test", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["hindi", "tamil", "telugu", "marathi", "bengali", "gujarati"],
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--sim-root", type=Path, default=SIM_ROOT)
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--force-resimulate", action="store_true")
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument("--trained-model-source", default=None)
    parser.add_argument("--trained-save-dir", default="./pretrained_models/ecapa_trained")
    parser.add_argument("--baseline-model-source", default=None)
    parser.add_argument("--baseline-save-dir", default=None)
    parser.add_argument("--cohort-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    from app.utils.windows_symlink_patch import apply_windows_symlink_fallback
    apply_windows_symlink_fallback()

    args = parse_args()
    settings = get_settings()
    args.reports_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    if not args.trained_model_source:
        write_blocked_report(
            args.reports_root,
            "missing_trained_model",
            "No trained ECAPA model source was provided. Pass --trained-model-source "
            "as a local SpeechBrain hparams directory or Hugging Face model id.",
        )
        return 2

    baseline_source = args.baseline_model_source or settings.ECAPA_MODEL_SOURCE
    baseline_save_dir = args.baseline_save_dir or settings.ECAPA_MODEL_SAVE_DIR
    cohort_threshold = args.cohort_threshold
    if cohort_threshold is None:
        cohort_threshold = settings.SIMILARITY_THRESHOLD

    try:
        manifest = prepare_kathbath_selection(args)
        write_overlap_confirmation(manifest, args.reports_root)
        simulated_manifest = simulate_kathbath_selection(manifest, args)
        trials = build_trial_pairs(simulated_manifest, args)
        write_trial_pairs(trials, args.reports_root)
        scores = score_trials(
            trials,
            {
                "baseline": (baseline_source, baseline_save_dir),
                "trained": (args.trained_model_source, args.trained_save_dir),
            },
            args,
        )
        write_scores(scores, args.reports_root)
        write_eer_report(scores, args.reports_root)
        write_det_plot(scores, args.reports_root)
        write_cohort_far(scores, args.reports_root, cohort_threshold)
        write_threshold_recommendation(scores, args.reports_root)
    except Exception as exc:  # noqa: BLE001
        write_blocked_report(args.reports_root, exc.__class__.__name__, str(exc))
        print(f"Part A stopped before producing final calibration artifacts: {exc}", file=sys.stderr)
        return 1

    return 0


def write_blocked_report(reports_root: Path, reason: str, detail: str) -> None:
    reports_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked",
        "reason": reason,
        "detail": detail,
        "no_thresholds_written": True,
        "config_modified": False,
    }
    (reports_root / "part_a_kathbath_blocked.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def prepare_kathbath_selection(args: argparse.Namespace) -> list[Utterance]:
    manifest_path = args.reports_root / KATHBATH_MANIFEST.name
    if manifest_path.exists() and not args.force_redownload:
        manifest = read_manifest(manifest_path)
        if has_required_selection(manifest, args.speaker_count, args.min_utterances):
            return manifest

    local_manifest = discover_local_kathbath(args.raw_root)
    if has_required_selection(local_manifest, args.speaker_count, args.min_utterances):
        selected = select_manifest_speakers(local_manifest, args.speaker_count, args.min_utterances)
        write_manifest(selected, manifest_path)
        return selected

    selected = download_kathbath_slice(args)
    write_manifest(selected, manifest_path)
    return selected


def has_required_selection(manifest: list[Utterance], speaker_count: int, min_utterances: int) -> bool:
    by_speaker = defaultdict(list)
    for item in manifest:
        by_speaker[item.speaker_key].append(item)
    qualified = [speaker for speaker, items in by_speaker.items() if len(items) >= min_utterances]
    return len(qualified) >= speaker_count


def discover_local_kathbath(root: Path) -> list[Utterance]:
    if not root.exists():
        return []
    manifest_path = root / "manifest.csv"
    if manifest_path.exists():
        return read_manifest(manifest_path)

    rows: list[Utterance] = []
    for wav_path in sorted(root.rglob("*.wav")):
        rel_parts = wav_path.relative_to(root).parts
        if len(rel_parts) < 3:
            continue
        lang = rel_parts[0]
        speaker_key = rel_parts[1]
        speaker_id = speaker_key
        gender = infer_gender_from_name(speaker_key) or infer_gender_from_name(wav_path.name)
        rows.append(
            Utterance(
                speaker_key=speaker_key,
                speaker_id=speaker_id,
                lang=lang,
                gender=gender,
                region="",
                utterance_index=len(rows),
                path=wav_path,
                source_fname=wav_path.name,
            )
        )
    return rows


def download_kathbath_slice(args: argparse.Namespace) -> list[Utterance]:
    try:
        import datasets.features.audio as fa
        fa.Audio.decode_example = lambda self, value, token_per_repo_id=None: value
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is present in this repo
        raise RuntimeError("The datasets package is required to download Kathbath") from exc

    pending: dict[str, list[tuple[dict, int]]] = defaultdict(list)
    selected_speakers: set[str] = set()
    utterances: list[Utterance] = []
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    for lang in args.languages:
        for split in ("valid", "train"):
            dataset = load_dataset(
                "ai4bharat/Kathbath",
                lang,
                split=split,
                streaming=True,
                token=token,
            )
            for row_index, row in enumerate(dataset):
                speaker_id = str(row.get("speaker_id", "")).strip()
                if not speaker_id:
                    continue
                gender = str(row.get("gender", "") or "").strip().lower()
                speaker_key = f"{lang}_{speaker_id}_{gender or 'unknown'}"
                if speaker_key in selected_speakers:
                    continue
                if len(selected_speakers) >= args.speaker_count:
                    break
                pending[speaker_key].append((row, row_index))
                if len(pending[speaker_key]) >= args.min_utterances:
                    selected_speakers.add(speaker_key)
                    utterances.extend(
                        materialize_kathbath_rows(
                            lang,
                            speaker_key,
                            speaker_id,
                            gender,
                            pending[speaker_key][: args.min_utterances],
                            args.raw_root,
                        )
                    )
            if len(selected_speakers) >= args.speaker_count:
                break
        if len(selected_speakers) >= args.speaker_count:
            break

    if len(selected_speakers) < args.speaker_count:
        raise RuntimeError(
            f"Only found {len(selected_speakers)} Kathbath speakers with "
            f"{args.min_utterances}+ utterances; requested {args.speaker_count}."
        )
    return sorted(utterances, key=lambda item: (item.speaker_key, item.utterance_index))


def materialize_kathbath_rows(
    lang: str,
    speaker_key: str,
    speaker_id: str,
    gender: str,
    rows: list[tuple[dict, int]],
    raw_root: Path,
) -> list[Utterance]:
    output: list[Utterance] = []
    for utterance_index, (row, row_index) in enumerate(rows):
        fname = str(row.get("fname") or f"{speaker_key}_{row_index}.wav")
        safe_name = sanitize_filename(Path(fname).stem) + ".wav"
        path = raw_root / lang / speaker_key / safe_name
        audio_field = row.get("audio_filepath")
        audio, sr = audio_from_hf_field(audio_field)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio.astype(np.float32), sr)
        output.append(
            Utterance(
                speaker_key=speaker_key,
                speaker_id=speaker_id,
                lang=lang,
                gender=gender,
                region=str(row.get("region", "") or ""),
                utterance_index=utterance_index,
                path=path,
                source_fname=fname,
            )
        )
    write_manifest(discover_local_kathbath(raw_root), raw_root / "manifest.csv")
    return output


def audio_from_hf_field(audio_field) -> tuple[np.ndarray, int]:
    import io
    if isinstance(audio_field, dict):
        if audio_field.get("array") is not None:
            return np.asarray(audio_field["array"], dtype=np.float32), int(audio_field["sampling_rate"])
        if audio_field.get("bytes") is not None:
            audio, sr = sf.read(io.BytesIO(audio_field["bytes"]), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), int(sr)
        if audio_field.get("path"):
            audio, sr = sf.read(str(audio_field["path"]), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), int(sr)
    if isinstance(audio_field, (str, Path)):
        audio, sr = sf.read(str(audio_field), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), int(sr)
    raise RuntimeError("Kathbath audio field did not contain decodable audio")


def select_manifest_speakers(
    manifest: list[Utterance], speaker_count: int, min_utterances: int
) -> list[Utterance]:
    by_speaker = defaultdict(list)
    for item in manifest:
        by_speaker[item.speaker_key].append(item)
    selected: list[Utterance] = []
    for speaker_key in sorted(by_speaker):
        items = sorted(by_speaker[speaker_key], key=lambda item: item.utterance_index)
        if len(items) >= min_utterances:
            selected.extend(items[:min_utterances])
        if len({item.speaker_key for item in selected}) >= speaker_count:
            break
    return selected


def write_overlap_confirmation(manifest: list[Utterance], reports_root: Path) -> None:
    kathbath_ids = {item.speaker_key for item in manifest}
    svarah_dirs = set()
    for root in [DATA_ROOT / "raw_svarah" / "wavs", REPO_ROOT / "data" / "svarah"]:
        if root.exists():
            svarah_dirs.update(path.name for path in root.iterdir() if path.is_dir())
    overlap = sorted(kathbath_ids & svarah_dirs)
    metadata_fields = sorted({"lang", "gender", "speaker_id", "region"} - {"region"})
    if any(item.region for item in manifest):
        metadata_fields.append("region")

    lines = [
        "Kathbath no-overlap check",
        f"Kathbath selected speakers: {len(kathbath_ids)}",
        f"Svarah speaker identifiers found locally: {len(svarah_dirs)}",
        f"Exact identifier overlaps: {len(overlap)}",
        "Result: zero exact speaker identifier overlap." if not overlap else "Result: overlap detected.",
        "",
        "Available Kathbath speaker metadata fields in this run: " + ", ".join(metadata_fields),
        "Note: this checks the speaker identifiers available in local artifacts. "
        "Kathbath does not expose real names in the dataset rows used here.",
    ]
    if overlap:
        lines.append("Overlapping identifiers: " + ", ".join(overlap))
    (reports_root / "kathbath_no_overlap_confirmation.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def simulate_kathbath_selection(manifest: list[Utterance], args: argparse.Namespace) -> list[Utterance]:
    noise_files = sorted((DATA_ROOT / "musan_noise").glob("*.wav"))
    output: list[Utterance] = []
    for item in manifest:
        for condition in CONDITIONS:
            destination = (
                args.sim_root
                / condition.name
                / item.lang
                / item.speaker_key
                / item.path.name
            )
            if args.force_resimulate or not destination.exists():
                simulate_file(
                    item.path,
                    destination,
                    condition.name,
                    noise_files,
                    seed=stable_seed(args.seed, condition.name, item.speaker_key, item.path.name),
                )
            output.append(
                Utterance(
                    speaker_key=item.speaker_key,
                    speaker_id=item.speaker_id,
                    lang=item.lang,
                    gender=item.gender,
                    region=item.region,
                    utterance_index=item.utterance_index,
                    path=destination,
                    source_fname=item.source_fname,
                )
            )
    write_manifest(output, args.reports_root / "kathbath_simulated_manifest.csv")
    return output


def build_trial_pairs(manifest: list[Utterance], args: argparse.Namespace) -> list[dict[str, str]]:
    clean_by_speaker: dict[str, list[Utterance]] = defaultdict(list)
    tests_by_condition_speaker: dict[tuple[str, str], list[Utterance]] = defaultdict(list)
    metadata: dict[str, Utterance] = {}

    for item in manifest:
        condition = item.path.parts[-4]
        metadata[item.speaker_key] = item
        if condition == "clean" and item.utterance_index < 5:
            clean_by_speaker[item.speaker_key].append(item)
        elif item.utterance_index >= 5:
            tests_by_condition_speaker[(condition, item.speaker_key)].append(item)

    speakers = sorted(clean_by_speaker)
    rng = random.Random(args.seed)
    trials: list[dict[str, str]] = []

    for condition in [item.name for item in CONDITIONS]:
        for target_speaker in speakers:
            tests = sorted(
                tests_by_condition_speaker.get((condition, target_speaker), []),
                key=lambda item: item.utterance_index,
            )
            for test in tests:
                trials.append(trial_row(condition, "genuine", target_speaker, target_speaker, test, metadata))

                same_cohort = [
                    speaker
                    for speaker in speakers
                    if speaker != target_speaker
                    and metadata[speaker].cohort_key == metadata[target_speaker].cohort_key
                ]
                other_speakers = [speaker for speaker in speakers if speaker != target_speaker]
                rng.shuffle(same_cohort)
                rng.shuffle(other_speakers)

                impostors = same_cohort[: args.same_cohort_impostors_per_test]
                for speaker in other_speakers:
                    if len(impostors) >= args.impostors_per_test:
                        break
                    if speaker not in impostors:
                        impostors.append(speaker)

                for impostor in impostors:
                    trials.append(trial_row(condition, "impostor", impostor, target_speaker, test, metadata))

    return trials


def trial_row(
    condition: str,
    trial_type: str,
    claimed_speaker: str,
    test_speaker: str,
    test: Utterance,
    metadata: dict[str, Utterance],
) -> dict[str, str]:
    claimed_meta = metadata[claimed_speaker]
    test_meta = metadata[test_speaker]
    return {
        "trial_id": stable_id(condition, trial_type, claimed_speaker, test_speaker, str(test.path)),
        "condition": condition,
        "trial_type": trial_type,
        "claimed_speaker": claimed_speaker,
        "test_speaker": test_speaker,
        "test_path": str(test.path),
        "claimed_cohort": claimed_meta.cohort_key,
        "test_cohort": test_meta.cohort_key,
        "is_same_cohort": str(claimed_meta.cohort_key == test_meta.cohort_key).lower(),
    }


def score_trials(
    trials: list[dict[str, str]],
    models: dict[str, tuple[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, str | float]]:
    enrollment_paths = collect_enrollment_paths(args.sim_root)
    all_audio_paths = sorted(
        {path for paths in enrollment_paths.values() for path in paths}
        | {Path(row["test_path"]) for row in trials}
    )
    results: list[dict[str, str | float]] = []

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

        start = time.time()
        for index, row in enumerate(trials, start=1):
            test_embedding = file_embeddings[Path(row["test_path"])]
            enrolled = enrollments[row["claimed_speaker"]]
            score = ECAPAService.cosine_similarity(test_embedding, enrolled)
            out = dict(row)
            out["model"] = model_name
            out["score"] = score
            results.append(out)
            if index % 500 == 0:
                elapsed = max(time.time() - start, 0.001)
                print(f"{model_name}: scored {index}/{len(trials)} trials at {index / elapsed:.1f}/sec")
    return results


def collect_enrollment_paths(sim_root: Path) -> dict[str, list[Path]]:
    by_speaker: dict[str, list[Path]] = defaultdict(list)
    clean_root = sim_root / "clean"
    for wav_path in sorted(clean_root.rglob("*.wav")):
        speaker_key = wav_path.parent.name
        by_speaker[speaker_key].append(wav_path)
    return {speaker: paths[:5] for speaker, paths in by_speaker.items() if len(paths) >= 5}


def cached_embedding(
    service: ECAPAService,
    path: Path,
    cache_root: Path,
    force: bool,
) -> np.ndarray:
    cache_path = cache_root / (stable_id(str(path), str(path.stat().st_mtime_ns)) + ".npy")
    if cache_path.exists() and not force:
        return np.load(cache_path)
    embedding = service.extract_embedding(str(path))
    np.save(cache_path, embedding)
    return embedding


def write_scores(scores: list[dict[str, str | float]], reports_root: Path) -> None:
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
        "score",
    ]
    write_dict_csv(reports_root / ALL_SCORES.name, scores, fieldnames)


def write_eer_report(scores: list[dict[str, str | float]], reports_root: Path) -> None:
    rows = []
    for model in sorted({str(row["model"]) for row in scores}):
        for condition in [item.name for item in CONDITIONS]:
            genuine = score_values(scores, model, condition, "genuine")
            impostor = score_values(scores, model, condition, "impostor")
            result = compute_eer(genuine, impostor)
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "eer_percent": f"{result.eer_percent:.6f}",
                    "threshold_at_eer": f"{result.threshold:.6f}",
                }
            )
    write_dict_csv(
        reports_root / "eer_report_kathbath.csv",
        rows,
        ["model", "condition", "eer_percent", "threshold_at_eer"],
    )


def write_det_plot(scores: list[dict[str, str | float]], reports_root: Path) -> None:
    plt.figure(figsize=(9, 7))
    for model in sorted({str(row["model"]) for row in scores}):
        for condition in [item.name for item in CONDITIONS]:
            genuine = score_values(scores, model, condition, "genuine")
            impostor = score_values(scores, model, condition, "impostor")
            far, frr = det_points(genuine, impostor)
            plt.plot(far * 100.0, frr * 100.0, label=f"{model} {condition}")
    plt.xlabel("False accept rate (%)")
    plt.ylabel("False reject rate (%)")
    plt.title("Kathbath DET Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(reports_root / "det_curve_kathbath.png", dpi=160)
    plt.close()


def write_cohort_far(scores: list[dict[str, str | float]], reports_root: Path, threshold: float) -> None:
    rows = []
    far_by_model_condition = {}
    for model in sorted({str(row["model"]) for row in scores}):
        for condition in [item.name for item in CONDITIONS]:
            cohort_rows = [
                row
                for row in scores
                if row["model"] == model
                and row["condition"] == condition
                and row["trial_type"] == "impostor"
                and str(row["is_same_cohort"]).lower() == "true"
            ]
            total = len(cohort_rows)
            false_accepts = sum(float(row["score"]) >= threshold for row in cohort_rows)
            ci_low, ci_high = wilson_interval(false_accepts, total)
            far = false_accepts / total if total else 0.0
            far_by_model_condition[(model, condition)] = far
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "threshold": f"{threshold:.6f}",
                    "same_cohort_impostor_trials": total,
                    "false_accepts": false_accepts,
                    "far_percent": f"{far * 100.0:.6f}",
                    "ci95_low_percent": f"{ci_low * 100.0:.6f}",
                    "ci95_high_percent": f"{ci_high * 100.0:.6f}",
                    "trained_worse_than_baseline": "",
                }
            )

    baseline = {
        condition: far
        for (model, condition), far in far_by_model_condition.items()
        if model == "baseline"
    }
    for row in rows:
        if row["model"] == "trained":
            row["trained_worse_than_baseline"] = str(
                far_by_model_condition[("trained", row["condition"])] > baseline.get(row["condition"], 1.0)
            ).lower()

    write_dict_csv(
        reports_root / "cohort_far_kathbath.csv",
        rows,
        [
            "model",
            "condition",
            "threshold",
            "same_cohort_impostor_trials",
            "false_accepts",
            "far_percent",
            "ci95_low_percent",
            "ci95_high_percent",
            "trained_worse_than_baseline",
        ],
    )


def write_threshold_recommendation(scores: list[dict[str, str | float]], reports_root: Path) -> None:
    trained = [row for row in scores if row["model"] == "trained"]
    if not trained:
        raise RuntimeError("No trained-model scores are available for threshold recommendation")

    all_impostor = [float(row["score"]) for row in trained if row["trial_type"] == "impostor"]
    same_cohort_impostor = [
        float(row["score"])
        for row in trained
        if row["trial_type"] == "impostor" and str(row["is_same_cohort"]).lower() == "true"
    ]
    clean_genuine = score_values(trained, "trained", "clean", "genuine")
    harsh_genuine = [
        float(row["score"])
        for row in trained
        if row["trial_type"] == "genuine"
        and row["condition"] in {"telephone_moderate", "telephone_severe"}
    ]
    moderate_genuine = score_values(trained, "trained", "telephone_moderate", "genuine")
    moderate_impostor = score_values(trained, "trained", "telephone_moderate", "impostor")
    moderate_eer = compute_eer(moderate_genuine, moderate_impostor)

    impostor_max = max(all_impostor)
    pass_margin = min(0.02, max(0.001, 1.0 - impostor_max))
    pass_threshold = min(1.0, impostor_max + pass_margin)
    moderate_genuine_p5 = percentile(moderate_genuine, 5)
    step_up_lower_bound = max(moderate_eer.threshold, moderate_genuine_p5)
    harsh_genuine_p1 = percentile(harsh_genuine, 1)
    hard_fail_margin = 0.001
    hard_fail_threshold = min(min(harsh_genuine) - hard_fail_margin, harsh_genuine_p1 - hard_fail_margin)

    clean_below_pass = sum(score < pass_threshold for score in clean_genuine)
    pass_friction = clean_below_pass / len(clean_genuine)
    same_cohort_false_accepts = sum(score >= pass_threshold for score in same_cohort_impostor)
    same_cohort_far_at_pass = (
        same_cohort_false_accepts / len(same_cohort_impostor) if same_cohort_impostor else 0.0
    )

    eers = {}
    for condition in [item.name for item in CONDITIONS]:
        result = compute_eer(
            score_values(trained, "trained", condition, "genuine"),
            score_values(trained, "trained", condition, "impostor"),
        )
        eers[condition] = {
            "eer_percent": result.eer_percent,
            "threshold_at_eer": result.threshold,
        }

    payload = {
        "source_artifacts": {
            "scores_csv": str(reports_root / ALL_SCORES.name),
            "eer_csv": str(reports_root / "eer_report_kathbath.csv"),
            "cohort_far_csv": str(reports_root / "cohort_far_kathbath.csv"),
            "det_curve_png": str(reports_root / "det_curve_kathbath.png"),
        },
        "impostor_distribution_all_conditions": {
            "max": impostor_max,
            "p95": percentile(all_impostor, 95),
            "p99": percentile(all_impostor, 99),
        },
        "same_cohort_impostor_distribution_all_conditions": {
            "max": max(same_cohort_impostor) if same_cohort_impostor else None,
            "p95": percentile(same_cohort_impostor, 95) if same_cohort_impostor else None,
            "p99": percentile(same_cohort_impostor, 99) if same_cohort_impostor else None,
        },
        "clean_genuine_distribution": {
            "p1": percentile(clean_genuine, 1),
            "p5": percentile(clean_genuine, 5),
            "p25": percentile(clean_genuine, 25),
            "p50": percentile(clean_genuine, 50),
        },
        "eer_crossover_points": eers,
        "recommended_thresholds": {
            "PASS_THRESHOLD": {
                "value": pass_threshold,
                "evidence": "Set above the observed maximum impostor score across all conditions.",
                "observed_impostor_max": impostor_max,
                "margin": pass_threshold - impostor_max,
            },
            "STEP_UP_LOWER_BOUND": {
                "value": step_up_lower_bound,
                "evidence": "Anchored to telephone_moderate EER and cross-checked against moderate genuine p5.",
                "telephone_moderate_eer_threshold": moderate_eer.threshold,
                "telephone_moderate_genuine_p5": moderate_genuine_p5,
            },
            "HARD_FAIL_THRESHOLD": {
                "value": hard_fail_threshold,
                "evidence": "Set below the observed minimum genuine score in telephone_moderate/severe.",
                "harsh_genuine_p1": harsh_genuine_p1,
                "harsh_genuine_min": min(harsh_genuine),
                "margin_below_min": min(harsh_genuine) - hard_fail_threshold,
            },
        },
        "clean_genuine_step_up_friction_at_pass_threshold": {
            "count_below_pass": clean_below_pass,
            "total_clean_genuine": len(clean_genuine),
            "percent": pass_friction * 100.0,
        },
        "same_cohort_far_at_pass_threshold": {
            "threshold": pass_threshold,
            "false_accepts": same_cohort_false_accepts,
            "total_same_cohort_impostors": len(same_cohort_impostor),
            "far_percent": same_cohort_far_at_pass * 100.0,
        },
        "config_modified": False,
    }
    (reports_root / "threshold_recommendation_kathbath.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def score_values(
    scores: Iterable[dict[str, str | float]],
    model: str,
    condition: str,
    trial_type: str,
) -> list[float]:
    return [
        float(row["score"])
        for row in scores
        if row["model"] == model and row["condition"] == condition and row["trial_type"] == trial_type
    ]


def read_manifest(path: Path) -> list[Utterance]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            Utterance(
                speaker_key=row["speaker_key"],
                speaker_id=row["speaker_id"],
                lang=row["lang"],
                gender=row.get("gender", ""),
                region=row.get("region", ""),
                utterance_index=int(row["utterance_index"]),
                path=Path(row["path"]),
                source_fname=row.get("source_fname", Path(row["path"]).name),
            )
            for row in csv.DictReader(handle)
        ]


def write_manifest(rows: list[Utterance], path: Path) -> None:
    fieldnames = [
        "speaker_key",
        "speaker_id",
        "lang",
        "gender",
        "region",
        "utterance_index",
        "path",
        "source_fname",
    ]
    write_dict_csv(
        path,
        [
            {
                "speaker_key": row.speaker_key,
                "speaker_id": row.speaker_id,
                "lang": row.lang,
                "gender": row.gender,
                "region": row.region,
                "utterance_index": row.utterance_index,
                "path": str(row.path),
                "source_fname": row.source_fname,
            }
            for row in rows
        ],
        fieldnames,
    )


def write_trial_pairs(trials: list[dict[str, str]], reports_root: Path) -> None:
    write_dict_csv(
        reports_root / TRIAL_PAIRS.name,
        trials,
        [
            "trial_id",
            "condition",
            "trial_type",
            "claimed_speaker",
            "test_speaker",
            "test_path",
            "claimed_cohort",
            "test_cohort",
            "is_same_cohort",
        ],
    )


def write_dict_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_gender_from_name(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith("_f") or lowered.endswith("-f") or "_f_" in lowered:
        return "f"
    if lowered.endswith("_m") or lowered.endswith("-m") or "_m_" in lowered:
        return "m"
    return ""


def sanitize_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return cleaned[:120] or "audio"


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def stable_seed(seed: int, *parts: str) -> int:
    return int(stable_id(str(seed), *parts)[:8], 16)


if __name__ == "__main__":
    raise SystemExit(main())
