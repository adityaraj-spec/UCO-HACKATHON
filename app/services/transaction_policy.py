"""
Risk-tier policy for transaction-aware voice verification.

The defaults live here instead of app/core/config.py because the project plan
marked the new config values as review-only. Production can still override
them by adding matching settings later.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.core.config import get_settings

settings = get_settings()

DEFAULT_MEDIUM_RISK_AMOUNT_THRESHOLD = Decimal("25000")
DEFAULT_HIGH_RISK_SNORM_PASS_THRESHOLD = 4.0

DECISION_OTP_REQUIRED = "otp_required"
DECISION_PHRASE_MISMATCH = "phrase_mismatch"
TIER_LOW = "LOW"
TIER_MEDIUM = "MEDIUM"
TIER_HIGH = "HIGH"

READ_ONLY_TRANSACTION_TYPES = {
    "balance_inquiry",
    "mini_statement",
    "account_statement",
    "statement_download",
}
HIGH_RISK_TRANSACTION_TYPES = {"add_payee", "limit_change"}
MONEY_MOVING_TRANSACTION_TYPES = {"fund_transfer", "transfer"}


@dataclass(frozen=True)
class TransactionContext:
    transaction_type: str | None = None
    transaction_amount: Decimal | None = None
    expected_phrase: str | None = None

    @property
    def has_transaction(self) -> bool:
        return bool(self.transaction_type or self.transaction_amount or self.expected_phrase)


@dataclass(frozen=True)
class TransactionDecision:
    tier: str | None
    decision: str
    verified: bool
    biometric_verified: bool
    otp_required: bool
    final_authorized: bool
    phrase_match: bool | None


def _medium_risk_amount_threshold() -> Decimal:
    value = getattr(settings, "TRANSACTION_MEDIUM_RISK_AMOUNT_THRESHOLD", None)
    if value is None:
        return DEFAULT_MEDIUM_RISK_AMOUNT_THRESHOLD
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return DEFAULT_MEDIUM_RISK_AMOUNT_THRESHOLD


def high_risk_snorm_pass_threshold() -> float:
    value = getattr(settings, "SNORM_HIGH_RISK_PASS_THRESHOLD", None)
    if value is None:
        return DEFAULT_HIGH_RISK_SNORM_PASS_THRESHOLD
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_HIGH_RISK_SNORM_PASS_THRESHOLD


def parse_transaction_context(
    transaction_type: str | None,
    transaction_amount: float | str | None,
    expected_phrase: str | None,
) -> TransactionContext:
    clean_type = transaction_type.strip().lower() if transaction_type else None
    amount: Decimal | None = None
    if transaction_amount not in (None, ""):
        try:
            amount = Decimal(str(transaction_amount))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("transaction_amount must be a valid number.") from exc
    phrase = expected_phrase.strip() if expected_phrase else None
    return TransactionContext(clean_type, amount, phrase)


def resolve_transaction_tier(context: TransactionContext) -> str | None:
    if not context.has_transaction:
        return None

    transaction_type = context.transaction_type or ""
    if transaction_type in HIGH_RISK_TRANSACTION_TYPES:
        return TIER_HIGH

    if transaction_type in MONEY_MOVING_TRANSACTION_TYPES:
        threshold = _medium_risk_amount_threshold()
        if context.transaction_amount is not None and context.transaction_amount >= threshold:
            return TIER_HIGH
        return TIER_MEDIUM

    if transaction_type in READ_ONLY_TRANSACTION_TYPES:
        return TIER_LOW

    return TIER_MEDIUM


def tier_requires_phrase(tier: str | None) -> bool:
    return tier in {TIER_MEDIUM, TIER_HIGH}


def apply_transaction_policy(
    *,
    base_decision: str,
    base_verified: bool,
    snorm_score: float,
    tier: str | None,
    phrase_match: bool | None,
) -> TransactionDecision:
    biometric_verified = base_verified

    if tier is None or tier == TIER_LOW:
        return TransactionDecision(
            tier=tier,
            decision=base_decision,
            verified=base_verified,
            biometric_verified=biometric_verified,
            otp_required=False,
            final_authorized=base_verified,
            phrase_match=phrase_match,
        )

    if tier_requires_phrase(tier) and phrase_match is not True:
        return TransactionDecision(
            tier=tier,
            decision=DECISION_PHRASE_MISMATCH,
            verified=False,
            biometric_verified=False,
            otp_required=False,
            final_authorized=False,
            phrase_match=False,
        )

    if tier == TIER_MEDIUM:
        return TransactionDecision(
            tier=tier,
            decision=base_decision,
            verified=base_verified,
            biometric_verified=biometric_verified,
            otp_required=False,
            final_authorized=base_verified,
            phrase_match=phrase_match,
        )

    if snorm_score >= high_risk_snorm_pass_threshold():
        return TransactionDecision(
            tier=tier,
            decision=DECISION_OTP_REQUIRED,
            verified=False,
            biometric_verified=True,
            otp_required=True,
            final_authorized=False,
            phrase_match=phrase_match,
        )

    if base_verified:
        return TransactionDecision(
            tier=tier,
            decision="step_up",
            verified=False,
            biometric_verified=True,
            otp_required=True,
            final_authorized=False,
            phrase_match=phrase_match,
        )

    return TransactionDecision(
        tier=tier,
        decision=base_decision,
        verified=False,
        biometric_verified=False,
        otp_required=False,
        final_authorized=False,
        phrase_match=phrase_match,
    )
