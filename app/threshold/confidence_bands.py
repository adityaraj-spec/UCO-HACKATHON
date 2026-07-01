"""
app/threshold/confidence_bands.py

Classification of similarity scores into decision confidence bands.

Three bands:
  1. PASS:       similarity >= threshold
  2. STEP_UP:    (threshold - STEP_UP_LOWER_BOUND) <= similarity < threshold
                 Triggers secondary factor authentication validation (OTP/PIN).
  3. HARD_FAIL:  similarity < (threshold - STEP_UP_LOWER_BOUND)
                 Direct lockout and fraud system alert.
"""

from __future__ import annotations

from enum import Enum


class DecisionBand(str, Enum):
    PASS = "PASS"
    STEP_UP = "STEP_UP"
    HARD_FAIL = "HARD_FAIL"


def classify_score(similarity: float, threshold: float, step_up_bound: float = 0.05) -> tuple[DecisionBand, str]:
    """
    Classify a cosine similarity score into a decision band.

    Args:
        similarity: Measured cosine similarity (0.0-1.0)
        threshold: Current adaptive threshold
        step_up_bound: Width of the step-up verification zone

    Returns:
        Tuple of (DecisionBand, description_reason)
    """
    step_up_threshold = threshold - step_up_bound

    if similarity >= threshold:
        return DecisionBand.PASS, f"Score {similarity:.3f} >= threshold {threshold:.3f}"
    
    if similarity >= step_up_threshold:
        return (
            DecisionBand.STEP_UP,
            f"Score {similarity:.3f} falls in step-up band [{step_up_threshold:.3f}, {threshold:.3f})"
        )

    return DecisionBand.HARD_FAIL, f"Score {similarity:.3f} < step_up threshold {step_up_threshold:.3f}"
