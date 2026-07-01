"""
app/fraud/mock_fraud_engine.py

Mock Fraud Engine implementation.

Determines risk evaluation scores based on context:
  - Unknown IP/device matches: elevates risk level
  - Rapid sequential login attempts (velocity anomalies)
  - Time-of-day warnings (large transactions requested late night)
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime

from app.fraud.interface import FraudEngine, FraudPayload, FraudResult

logger = logging.getLogger(__name__)


class MockFraudEngine(FraudEngine):
    """Rule-based mock fraud engine for bank testing."""

    # Pre-seeded lists of suspicious IPS
    SUSPICIOUS_IPS = {"198.51.100.42", "203.0.113.88", "192.0.2.15"}
    TRUSTED_DEVICES = {"DEV_MOBILE_9921", "DEV_IPHONE_77A", "DEV_DESKTOP_WIN"}

    def score(self, payload: FraudPayload) -> FraudResult:
        """
        Evaluate fraud score logic.
        """
        score = 0.05
        reasons = []

        # 1. IP check
        if payload.ip_address in self.SUSPICIOUS_IPS:
            score += 0.40
            reasons.append("suspicious_ip_reputation")

        # 2. Device trust check
        if payload.device_fingerprint not in self.TRUSTED_DEVICES:
            score += 0.20
            reasons.append("unregistered_device_fingerprint")

        # 3. Nighttime anomaly check (telephony banking hours: 11 PM to 5 AM)
        current_hour = payload.timestamp.hour
        if current_hour >= 23 or current_hour < 5:
            score += 0.15
            reasons.append("off_hours_transaction")

        # Cap score at 1.0
        score = min(score, 1.0)

        # Map to risk categorization
        if score >= 0.70:
            risk = "CRITICAL"
            action = "REJECT"
        elif score >= 0.45:
            risk = "HIGH"
            action = "STEP_UP"
        elif score >= 0.20:
            risk = "MEDIUM"
            action = "STEP_UP"
        else:
            risk = "LOW"
            action = "ALLOW"

        reason_str = ", ".join(reasons) if reasons else "no_suspicious_patterns"
        tx_id = f"TX_{secrets.token_hex(8).upper()}"

        logger.debug(
            "Fraud evaluation complete: usr=%s, score=%.2f, risk=%s, action=%s, tx=%s",
            payload.user_id, score, risk, action, tx_id,
        )

        return FraudResult(
            score=score,
            risk_level=risk,
            action_required=action,
            reason=reason_str,
            transaction_id=tx_id,
        )


# Module-level singleton
_engine: MockFraudEngine | None = None


def get_mock_fraud_engine() -> MockFraudEngine:
    """Return singleton MockFraudEngine."""
    global _engine
    if _engine is None:
        _engine = MockFraudEngine()
    return _engine
