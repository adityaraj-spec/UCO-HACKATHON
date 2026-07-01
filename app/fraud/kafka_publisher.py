"""
app/fraud/kafka_publisher.py

Kafka security event publisher.

Publishes critical authentication, threshold changes, and fraud events to
Kafka topic 'voice-auth-events' for downstream bank SIEM/Fraud platforms.
Falls back to a warning log if Kafka is unavailable or disabled.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class KafkaEventPublisher:
    """Publishes security and audit events to an Apache Kafka broker cluster."""

    def __init__(self) -> None:
        self.enabled = settings.KAFKA_ENABLED
        self.bootstrap_servers = settings.KAFKA_BOOTSTRAP_SERVERS
        self.topic = settings.KAFKA_TOPIC
        self._producer = None

        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        """Establish connection with the Kafka broker."""
        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=3000,
                retries=2,
            )
            logger.info("Kafka publisher connected to brokers: %s", self.bootstrap_servers)
        except Exception as e:
            logger.warning(
                "Failed to initialize Kafka producer (%s). Mock publisher will be used.",
                e
            )
            self.enabled = False

    def publish_event(self, event_type: str, payload: dict[str, Any]) -> bool:
        """
        Publish structured security event payload.

        Args:
            event_type: Category (e.g. AUTH_FAIL, BIO_DRIFT_ALERT, VELOCITY_ALERT)
            payload: Event data dictionary

        Returns:
            True if published successfully, False otherwise
        """
        event = {
            "event_type": event_type,
            "timestamp": payload.get("timestamp") or logging.Formatter.default_time_format,
            "data": payload,
        }

        if self.enabled and self._producer is not None:
            try:
                # Async send
                future = self._producer.send(self.topic, value=event)
                # Wait up to 1 second to confirm receipt (in banking, safety > async fire-&-forget)
                future.get(timeout=1.0)
                logger.debug("Kafka: published event %s to topic %s", event_type, self.topic)
                return True
            except Exception as e:
                logger.error("Kafka publish error for event %s: %s", event_type, e)
                return False

        # Fallback to local warning system when Kafka is offline
        logger.warning(
            "[MOCK KAFKA EVENT] Topic: %s | Event: %s | Data: %s",
            self.topic, event_type, json.dumps(payload)
        )
        return True


# Module-level singleton
_publisher: KafkaEventPublisher | None = None


def get_kafka_publisher() -> KafkaEventPublisher:
    """Return singleton KafkaEventPublisher."""
    global _publisher
    if _publisher is None:
        _publisher = KafkaEventPublisher()
    return _publisher
