from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class EERResult:
    eer_percent: float
    threshold: float
    far: float
    frr: float


def compute_eer(genuine_scores: list[float], impostor_scores: list[float]) -> EERResult:
    """Compute EER for similarity scores where larger means more likely genuine."""
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)

    if genuine.size == 0:
        raise ValueError("At least one genuine score is required")
    if impostor.size == 0:
        raise ValueError("At least one impostor score is required")

    thresholds = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.concatenate(
        [[np.nextafter(thresholds[0], -np.inf)], thresholds, [np.nextafter(thresholds[-1], np.inf)]]
    )

    best_threshold = float(thresholds[0])
    best_far = 1.0
    best_frr = 0.0
    best_gap = float("inf")
    best_mean = float("inf")

    for threshold in thresholds:
        far = float(np.mean(impostor >= threshold))
        frr = float(np.mean(genuine < threshold))
        gap = abs(far - frr)
        mean_error = (far + frr) / 2.0
        if gap < best_gap or (gap == best_gap and mean_error < best_mean):
            best_gap = gap
            best_mean = mean_error
            best_threshold = float(threshold)
            best_far = far
            best_frr = frr

    return EERResult(
        eer_percent=100.0 * ((best_far + best_frr) / 2.0),
        threshold=best_threshold,
        far=best_far,
        frr=best_frr,
    )


def det_points(genuine_scores: list[float], impostor_scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return FAR and FRR arrays for DET-style plots."""
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("Both genuine and impostor scores are required")

    thresholds = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.concatenate(
        [[np.nextafter(thresholds[0], -np.inf)], thresholds, [np.nextafter(thresholds[-1], np.inf)]]
    )

    far = np.array([np.mean(impostor >= threshold) for threshold in thresholds], dtype=np.float64)
    frr = np.array([np.mean(genuine < threshold) for threshold in thresholds], dtype=np.float64)
    return far, frr


def percentile(scores: list[float], value: float) -> float:
    if not scores:
        raise ValueError("Cannot compute percentile of an empty score list")
    return float(np.percentile(np.asarray(scores, dtype=np.float64), value))


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    if total == 0:
        return (0.0, 0.0)
    if confidence != 0.95:
        raise ValueError("Only 95 percent Wilson intervals are currently supported")

    z = 1.959963984540054
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return ((centre - margin) / denom, (centre + margin) / denom)
