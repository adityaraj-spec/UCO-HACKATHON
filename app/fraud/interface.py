"""
app/fraud/interface.py

Abstract interfaces for the external Fraud Engine and risk analysis systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FraudPayload:
    """Authentication request context passed to the fraud engine."""
    user_id: str
    ip_address: str
    device_fingerprint: str
    latitude: float | None = None
    longitude: float | None = None
    timestamp: datetime = datetime.now()


@dataclass
class FraudResult:
    """Fraud engine evaluation result."""
    score: float                # 0.0 (low risk) to 1.0 (critical high risk)
    risk_level: str             # LOW | MEDIUM | HIGH | CRITICAL
    action_required: str       # ALLOW | STEP_UP | REJECT
    reason: str
    transaction_id: str


class FraudEngine(ABC):
    """Abstract interface for the external Fraud Engine."""

    @abstractmethod
    def score(self, payload: FraudPayload) -> FraudResult:
        """Evaluate fraud score for a given payload context."""
        pass

