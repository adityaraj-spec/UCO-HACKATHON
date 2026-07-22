"""
app/services/circuit_breaker.py

Resilient circuit breaker for FAISS index operations.
Opens on 3 failures within 30s, falling back to PostgreSQL pgvector query. Half-opens after 10s.
"""

import time
from app.core.logging import get_logger

log = get_logger(__name__)


class FAISSCircuitBreaker:
    """Circuit breaker pattern wrapping FAISS index searches."""

    def __init__(self, max_failures: int = 3, failure_window_sec: float = 30.0, cooldown_sec: float = 10.0):
        self.max_failures = max_failures
        self.failure_window_sec = failure_window_sec
        self.cooldown_sec = cooldown_sec

        self.failures: list[float] = []
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.opened_at: float = 0.0

    def can_execute(self) -> bool:
        """Check if call can proceed through circuit breaker."""
        now = time.time()

        if self.state == "OPEN":
            if now - self.opened_at > self.cooldown_sec:
                log.info("FAISS Circuit Breaker transitioning to HALF-OPEN")
                self.state = "HALF-OPEN"
                return True
            return False

        return True

    def record_success(self) -> None:
        """Record successful execution."""
        if self.state == "HALF-OPEN":
            log.info("FAISS Circuit Breaker restored to CLOSED")
            self.state = "CLOSED"
            self.failures.clear()

    def record_failure(self) -> None:
        """Record execution failure."""
        now = time.time()
        self.failures.append(now)
        # Purge old failures
        self.failures = [t for t in self.failures if now - t <= self.failure_window_sec]

        if len(self.failures) >= self.max_failures and self.state != "OPEN":
            log.error("FAISS Circuit Breaker OPENED after %d failures", len(self.failures))
            self.state = "OPEN"
            self.opened_at = now
