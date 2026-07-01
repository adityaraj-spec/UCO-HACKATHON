"""
app/emergency/access_scope.py

Access permission scopes granted to emergency contacts.
"""

from enum import Enum


class EmergencyAccessScope(str, Enum):
    """
    Emergency account access permissions.

    - READ_ONLY: Check balances, beneficiary details, and statements. No transfers.
    - LIMITED_TRANSACTION: Verify transactions under ₹10,000 INR.
    - FULL_TRANSACTION: Full access to the customer account.
    """
    READ_ONLY = "READ_ONLY"
    LIMITED_TRANSACTION = "LIMITED_TRANSACTION"
    FULL_TRANSACTION = "FULL_TRANSACTION"
