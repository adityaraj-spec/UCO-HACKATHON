"""
app/threshold/engine.py

Adaptive Threshold Engine.

Dynamically calibrates verification similarity thresholds per user based on:
  1. Historical authentication performance:
     - Low SNR environment: slightly lower threshold (but keep a floor)
     - Clean, high-SNR environment: raise threshold (stricter check)
  2. Illness Tolerance Window:
     - Declines voice similarity slightly (e.g. by 0.05) if an illness window
       is active (marked by user or triggered after soft FRR/OTP lockout).
  3. Failure tracking:
     - Relaxes threshold incrementally on consecutive failures (e.g., -0.02)
       to avoid frustration, down to the floor limit, provided context
       (IP/device) is low-risk.
  4. Contextual Risk:
     - Increases threshold (e.g., +0.05) if fraud risk is elevated.

Maintains:
  - baseline: global system default (e.g. 0.65)
  - current: actual active threshold
  - floor: minimum allowed threshold (default 0.50)
  - ceiling: maximum allowed threshold (default 0.85)
"""

from __future__ import annotations

import logging
import numpy as np
from datetime import datetime, timezone

from app.core.config import get_settings
from app.models.layer2.user_threshold import UserThreshold

logger = logging.getLogger(__name__)
settings = get_settings()


class AdaptiveThresholdEngine:
    """Calculates and manages user-specific verification thresholds."""

    def __init__(self) -> None:
        self.default_baseline = settings.SIMILARITY_THRESHOLD
        self.step_up_lower_bound = settings.STEP_UP_LOWER_BOUND

    def calibrate_threshold(
        self,
        user_threshold: UserThreshold,
        consec_failures: int,
        illness_active: bool,
        fraud_risk_level: str,  # LOW | MEDIUM | HIGH | CRITICAL
        ip_device_trusted: bool,
        enrolled_via: str = "MOBILE_APP",
    ) -> float:
        """
        Compute effective similarity threshold for the user.

        Args:
            user_threshold: UserThreshold database model instance
            consec_failures: Count of consecutive failures
            illness_active: Flag indicating active illness window (sore throat, etc.)
            fraud_risk_level: Fraud engine risk category
            ip_device_trusted: Context authenticity evaluation
            enrolled_via: Dynamic channel validator (e.g. IVR, MOBILE_APP)

        Returns:
            Calculated target threshold (float)
        """
        # Start with user-specific baseline, defaulting to global config
        base = user_threshold.threshold_baseline or self.default_baseline
        eff = base

        reasons = []

        # 0. Adjust baseline if enrolled via a noisy/less-secure channel (e.g. IVR)
        if enrolled_via == "IVR":
            ivr_offset = 0.05
            eff += ivr_offset
            reasons.append(f"ivr_enrollment_channel_penalty(+{ivr_offset:.2f})")

        # 1. Apply consecutive failures relaxation (only if device/IP is trusted)
        # Rejects relaxing threshold for unknown/risky devices to prevent brute-force attacks
        if consec_failures > 0 and ip_device_trusted:
            relaxation = min(consec_failures * 0.02, 0.06)  # Cap at 0.06 relaxation
            eff -= relaxation
            reasons.append(f"consecutive_failures_relaxation(-{relaxation:.2f})")

        # 2. Apply illness tolerance window relaxation
        if illness_active:
            # Check expiration
            now = datetime.now(timezone.utc)
            if user_threshold.illness_window_expires_at and user_threshold.illness_window_expires_at > now:
                relaxation = 0.05
                eff -= relaxation
                reasons.append(f"illness_window_relaxation(-{relaxation:.2f})")
            else:
                user_threshold.illness_window_active = False
                user_threshold.illness_window_relaxation = 0.0

        # 3. Elevate threshold if external fraud risk is high/critical
        if fraud_risk_level in ("HIGH", "CRITICAL"):
            elevation = 0.05
            eff += elevation
            reasons.append(f"fraud_risk_elevation(+{elevation:.2f})")
        elif fraud_risk_level == "MEDIUM":
            elevation = 0.02
            eff += elevation
            reasons.append(f"medium_fraud_risk_elevation(+{elevation:.2f})")

        # 4. Bind threshold to absolute floor and ceiling limits
        floor = user_threshold.threshold_floor or 0.50
        ceiling = user_threshold.threshold_ceiling or 0.85

        final_thr = float(np.clip(eff, floor, ceiling))
        user_threshold.threshold_current = final_thr
        user_threshold.last_adjustment_reason = ", ".join(reasons) if reasons else "NO_ADJUSTMENT"

        logger.debug(
            "AdaptiveThreshold: user %s threshold calculated: %.3f (base=%.3f, details=%s)",
            user_threshold.user_id, final_thr, base, user_threshold.last_adjustment_reason,
        )

        return final_thr


# Module-level singleton
_threshold_engine: AdaptiveThresholdEngine | None = None


def get_threshold_engine() -> AdaptiveThresholdEngine:
    """Return singleton AdaptiveThresholdEngine."""
    global _threshold_engine
    if _threshold_engine is None:
        _threshold_engine = AdaptiveThresholdEngine()
    return _threshold_engine
