"""
app/fraud/kafka_consumer.py

Minimal Kafka consumer implementation for logging consumed security events in Layer 2.
Intended as an end-to-end example/demo for SIEM/fraud-analytics logging integration.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class KafkaEventConsumer:
    """Minimal consumer that subscribes to the Kafka event bus and logs messages."""

    def __init__(self) -> None:
        self.enabled = settings.KAFKA_ENABLED
        self.bootstrap_servers = settings.KAFKA_BOOTSTRAP_SERVERS
        self.topic = settings.KAFKA_TOPIC
        self._consumer = None
        self._thread = None
        self._running = False

    def start_listening(self, callback: Callable[[dict], None] | None = None) -> None:
        """Starts a background thread to consume messages if Kafka is enabled."""
        if not self.enabled:
            logger.info("Kafka consumer disabled by configuration.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, args=(callback,), daemon=True)
        self._thread.start()
        logger.info("Kafka consumer background listener started.")

    def stop_listening(self) -> None:
        """Gracefully stops the consumer thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Kafka consumer stopped.")

    def _run(self, callback: Callable[[dict], None] | None = None) -> None:
        try:
            from kafka import KafkaConsumer
            self._consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                consumer_timeout_ms=1000,
            )
            logger.info("Kafka consumer subscribed to topic %s", self.topic)
        except Exception as e:
            logger.error("Failed to initialize Kafka consumer connection: %s", e)
            self._running = False
            return

        while self._running:
            try:
                message_batch = self._consumer.poll(timeout_ms=500)
                for _topic_partition, records in message_batch.items():
                    for record in records:
                        logger.info("Received Kafka event: %s", record.value)
                        if callback:
                            try:
                                callback(record.value)
                            except Exception as cb_err:
                                logger.error("Callback error in Kafka consumer: %s", cb_err)
            except Exception as e:
                logger.error("Error polling Kafka events: %s", e)

        if self._consumer:
            self._consumer.close()


_consumer_instance: KafkaEventConsumer | None = None


def get_kafka_consumer() -> KafkaEventConsumer:
    """Return singleton KafkaEventConsumer."""
    global _consumer_instance
    if _consumer_instance is None:
        _consumer_instance = KafkaEventConsumer()
    return _consumer_instance
