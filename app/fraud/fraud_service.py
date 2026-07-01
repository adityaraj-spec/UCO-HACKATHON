"""
app/fraud/fraud_service.py

Fraud Integration Service.

Orchestrates context gathering, calls the pluggable Fraud Engine, and
dispatches security alert events to Kafka topic asynchronously.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import uuid

from app.fraud.interface import FraudPayload, FraudResult
from app.fraud.kafka_publisher import get_kafka_publisher
from app.fraud.mock_fraud_engine import get_mock_fraud_engine

logger = logging.getLogger(__name__)


class FraudService:
    """Orchestrates security profiling and threat metric updates."""

    def __init__(self) -> None:
        self.engine = get_mock_fraud_engine()
        self.publisher = get_kafka_publisher()

    def evaluate_risk(
        self,
        user_id: uuid.UUID,
        ip_address: str,
        device_fingerprint: str,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> FraudResult:
        """
        Evaluate full risk profile and dispatch event to Kakfa in a non-blocking step.
        """
        payload = FraudPayload(
            user_id=str(user_id),
            ip_address=ip_address,
            device_fingerprint=device_fingerprint,
            latitude=latitude,
            longitude=longitude,
            timestamp=datetime.now(timezone.utc),
        )

        # 1. Run inference on fraud engine
        result = self.engine.score(payload)

        # 2. Asynchronously fire security event to Kafka for bank SIEM
        event_data = {
            "user_id": str(user_id),
            "ip_address": ip_address,
            "device_fingerprint": device_fingerprint,
            "risk_level": result.risk_level,
            "score": result.score,
            "action": result.action_required,
            "reason": result.reason,
            "transaction_id": result.transaction_id,
            "timestamp": payload.timestamp.isoformat(),
        }

        # If FRAUD risk is HIGH or CRITICAL, publish alert
        if result.risk_level in ("HIGH", "CRITICAL"):
            self.publisher.publish_event("FRAUD_ALERT_HIGH", event_data)
        else:
            self.publisher.publish_event("FRAUD_CHECK_PASS", event_data)

        return result


# Module-level singleton
_fraud_service: FraudService | None = None


def get_fraud_service() -> FraudService:
    """Return singleton FraudService."""
    global _fraud_service
    if _fraud_service is None:
        _fraud_service = FraudService()
    return _fraud_service

