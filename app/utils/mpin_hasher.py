"""Hashing helpers for 6-digit MPIN authentication."""

from __future__ import annotations

import bcrypt


def hash_mpin(mpin: str) -> str:
    return bcrypt.hashpw(mpin.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_mpin(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ValueError:
        return False
