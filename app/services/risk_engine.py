"""
app/services/risk_engine.py

PhaseGuard Risk Engine.

Combines the output of Layer 1 (AI/deepfake voice detection probability)
and the raw Layer 2 BioHash similarity into a single risk assessment. This
risk level is separate from the live identity decision, which is made by
the verification service's s-norm path.

Decision rules (as specified by the PhaseGuard architecture):

    1. layer1_score > LAYER1_FRAUD_THRESHOLD (default 0.70)
            => risk_level = FRAUD_ALERT  (likely AI-generated voice)

    2. speaker_similarity < SIMILARITY_THRESHOLD
            => risk_level = FRAUD_ALERT  (low raw Layer 2 similarity)

    3. otherwise
            => risk_level = CLEAN

The numeric `risk_score` (0-100) is a continuous blend of both signals so
that the dashboard / history views can show *how* clean or risky a call
was, even when the categorical decision is the same.
"""

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.risk import RiskInput, RiskResult

settings = get_settings()
log = get_logger(__name__)

FRAUD_ALERT = "FRAUD_ALERT"
CLEAN = "CLEAN"


def _combined_score(layer1_score: float, speaker_similarity: float) -> float:
    """
    Compute a continuous 0-100 risk score blending both layers.

    - The Layer 1 contribution grows with the AI-voice probability.
    - The Layer 2 contribution grows the further the similarity falls below
      a perfect match (1.0).

    Both contributions are weighted equally (50/50) and combined into a
    single percentage.
    """
    layer1_component = max(0.0, min(1.0, layer1_score))

    # Map similarity (-1..1, practically 0..1) onto an "identity risk" in
    # 0..1, where 1.0 means a complete mismatch and 0.0 means a perfect
    # match.
    identity_risk = max(0.0, min(1.0, 1.0 - speaker_similarity))

    combined = (layer1_component * 0.5 + identity_risk * 0.5) * 100.0
    return round(combined, 2)


def compute_risk(layer1_score: float, speaker_similarity: float) -> RiskResult:
    """
    Evaluate the PhaseGuard risk rules for a single verification attempt.

    Args:
        layer1_score: AI-voice-detection probability from Layer 1, in [0, 1].
        speaker_similarity: Raw Layer 2 similarity, in [-1, 1].

    Returns:
        RiskResult(risk_score, risk_level)
    """
    risk_score = _combined_score(layer1_score, speaker_similarity)

    if layer1_score > settings.LAYER1_FRAUD_THRESHOLD:
        risk_level = FRAUD_ALERT
        log.info(
            f"Risk engine: FRAUD_ALERT - layer1_score={layer1_score:.4f} "
            f"exceeds threshold {settings.LAYER1_FRAUD_THRESHOLD}"
        )
    elif speaker_similarity < settings.SIMILARITY_THRESHOLD:
        risk_level = FRAUD_ALERT
        log.info(
            f"Risk engine: FRAUD_ALERT - speaker_similarity={speaker_similarity:.4f} "
            f"below threshold {settings.SIMILARITY_THRESHOLD}"
        )
    else:
        risk_level = CLEAN
        log.info(
            f"Risk engine: CLEAN - layer1_score={layer1_score:.4f}, "
            f"speaker_similarity={speaker_similarity:.4f}"
        )

    return RiskResult(risk_score=risk_score, risk_level=risk_level)


def compute_risk_from_input(risk_input: RiskInput) -> RiskResult:
    """Convenience wrapper accepting a `RiskInput` pydantic model."""
    return compute_risk(risk_input.layer1_score, risk_input.speaker_similarity)
