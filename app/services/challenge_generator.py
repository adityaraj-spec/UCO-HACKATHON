"""
Dynamic spoken-sentence challenge generator for Voice KYC.

Sentences are transaction-related banking phrases so voice enrollment and
liveness checks mimic actual voice banking transaction approvals.
"""

import secrets

TRANSACTION_ACTIONS = [
    "I authorize a fund transfer of",
    "Confirm payment of",
    "Approve transfer of",
    "Authorize instant money transfer of",
    "Confirm online bill payment of",
    "I request transfer of",
    "Verify UCO bank transaction of",
    "Authorize money transfer of",
]

TRANSACTION_TARGETS = [
    "rupees for transaction number",
    "rupees from my UCO account number",
    "rupees to payee account number",
    "rupees for reference number",
    "rupees under transaction number",
    "rupees from savings account number",
    "rupees for order number",
    "rupees to beneficiary number",
]

AMOUNTS = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 7500, 10000]


def generate_challenge_sentence() -> str:
    action = secrets.choice(TRANSACTION_ACTIONS)
    target = secrets.choice(TRANSACTION_TARGETS)
    amount = secrets.choice(AMOUNTS)
    number = secrets.randbelow(900) + 100
    return f"{action} {amount} {target} {number}."

