"""
app/utils/asr_verifier.py

Lightweight ASR transcript verifier for challenge-response liveness testing.
"""

import re
from app.core.logging import get_logger

log = get_logger(__name__)

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SCALE_WORDS = {
    "hundred": 100,
    "thousand": 1000,
    "lakh": 100000,
    "lakhs": 100000,
    "lac": 100000,
    "lacs": 100000,
    "crore": 10000000,
    "crores": 10000000,
}


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation from text."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def verify_spoken_challenge(spoken_text: str | None, expected_phrase: str) -> bool:
    """
    Verify if the spoken text matches the expected challenge text phrase.
    Returns True if match, False otherwise.
    """
    if not spoken_text:
        log.warning("Spoken text missing for challenge verification")
        return False

    norm_spoken = normalize_text(spoken_text)
    norm_expected = normalize_text(expected_phrase)

    if norm_expected in norm_spoken or norm_spoken in norm_expected:
        return True

    # Check word overlap ratio
    spoken_words = set(norm_spoken.split())
    expected_words = set(norm_expected.split())
    if not expected_words:
        return False

    overlap = len(spoken_words.intersection(expected_words)) / len(expected_words)
    return overlap >= 0.6


def _numbers_from_words(text: str) -> set[str]:
    numbers: set[str] = set()
    current = 0
    active = False

    for token in normalize_text(text).split():
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            active = True
        elif token in SCALE_WORDS and active:
            scale = SCALE_WORDS[token]
            current = max(current, 1) * scale
            numbers.add(str(current))
            current = 0
            active = False
        else:
            if active:
                numbers.add(str(current))
                current = 0
                active = False

    if active:
        numbers.add(str(current))

    return numbers


def _numeric_tokens(text: str) -> set[str]:
    compact = normalize_text(text).replace(" ", "")
    digits = {match.lstrip("0") or "0" for match in re.findall(r"\d+", compact)}
    return digits | _numbers_from_words(text)


def verify_transaction_phrase(spoken_text: str | None, expected_phrase: str) -> bool:
    """
    Fuzzy-match a transaction phrase transcript against the expected prompt.

    Amount/account substrings are treated as hard anchors when present. This
    keeps "five thousand" and "5000" equivalent without adding an ASR package.
    """
    if not spoken_text:
        log.warning("Spoken text missing for transaction phrase verification")
        return False

    expected_numbers = _numeric_tokens(expected_phrase)
    spoken_numbers = _numeric_tokens(spoken_text)
    if expected_numbers and not expected_numbers.issubset(spoken_numbers):
        return False

    return verify_spoken_challenge(spoken_text, expected_phrase)
