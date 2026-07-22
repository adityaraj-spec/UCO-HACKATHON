"""
Dynamic spoken-sentence challenge generator for Voice KYC.

Sentences are short and banking-themed so ASR has a fair target while each
challenge remains fresh enough to make pre-recorded replay impractical.
"""

import secrets

SUBJECTS = [
    "my banking assistant",
    "the secure branch",
    "our account officer",
    "the mobile teller",
    "my UCO account",
    "the customer desk",
]
VERBS = [
    "confirms",
    "verified",
    "recorded",
    "approved",
    "checked",
    "reported",
]
OBJECTS = [
    "a safe login",
    "the pending request",
    "a valid session",
    "the account update",
    "a stable transaction",
    "the voice check",
]


def generate_challenge_sentence() -> str:
    subject = secrets.choice(SUBJECTS)
    verb = secrets.choice(VERBS)
    obj = secrets.choice(OBJECTS)
    number = secrets.randbelow(900) + 100
    return f"{subject.capitalize()} {verb} {obj} number {number}."
