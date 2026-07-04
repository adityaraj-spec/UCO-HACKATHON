"""
app/emergency/cbs_provider.py

Core Banking System (CBS) integration provider Strategy pattern.
Defines public interface for validating branch override capability in Finacle CBS.
"""

from __future__ import annotations

import abc
import logging

logger = logging.getLogger(__name__)


class CBSProviderBase(abc.ABC):
    """Abstract Base Class for Core Banking System (Finacle) integrations."""

    @abc.abstractmethod
    async def verify_override_eligibility(self, account_number: str) -> bool:
        """Returns True if the target account is in good standing and permits biometric overrides."""
        pass

    @abc.abstractmethod
    async def register_biometric_override_flag(
        self, account_number: str, scope: str, actor_id: str
    ) -> bool:
        """Register the bypass event in the Finacle audit ledger."""
        pass

    @abc.abstractmethod
    async def validate_branch_code(self, branch_code: str) -> bool:
        """Verify the branch code is valid and active within the bank network."""
        pass


class MockCBSProvider(CBSProviderBase):
    """Mock implementation of Core Banking System integration for testing/demos."""

    def __init__(self, simulate_failure: bool = False) -> None:
        self.simulate_failure = simulate_failure

    async def verify_override_eligibility(self, account_number: str) -> bool:
        logger.info("[MockCBS] Verifying override eligibility for account: %s", account_number)
        if self.simulate_failure or account_number == "9999999999":
            return False
        return True

    async def register_biometric_override_flag(
        self, account_number: str, scope: str, actor_id: str
    ) -> bool:
        logger.info(
            "[MockCBS] Flagging override for account %s, scope: %s, actor: %s",
            account_number,
            scope,
            actor_id,
        )
        return True

    async def validate_branch_code(self, branch_code: str) -> bool:
        logger.info("[MockCBS] Validating branch code: %s", branch_code)
        if self.simulate_failure or branch_code == "BAD_BRANCH":
            return False
        # Simulates typical 3-4 digit branch code validation
        return len(branch_code) >= 3


_cbs_provider: CBSProviderBase | None = None


def get_cbs_provider() -> CBSProviderBase:
    """Return singleton CBSProvider."""
    global _cbs_provider
    if _cbs_provider is None:
        _cbs_provider = MockCBSProvider()
    return _cbs_provider
