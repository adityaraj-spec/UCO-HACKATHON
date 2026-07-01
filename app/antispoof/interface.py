"""
app/antispoof/interface.py

Abstract interface for the Anti-Spoofing and Liveness Detection Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SpoofResult:
    """Detailed result of an anti-spoofing check."""

    is_spoof: bool
    confidence: float          # Probability that the sample is a spoof (0.0-1.0)
    method_detected: str       # REPLAY | SYNTHESIS_TTS | DEEPFAKE_CLONE | NONE
    rejection_reason: str | None


class AntiSpoofEngine(ABC):
    """
    Interface for anti-spoofing and liveness engines.

    To support pluggable architecture, any deepfake or replay detection model
    (e.g., AASIST, RawBoost, ResMax) must implement this interface.
    """

    @abstractmethod
    def analyze_liveness(
        self,
        waveform_bytes: bytes,
        claimed_phrase: str | None = None,
    ) -> SpoofResult:
        """
        Analyze audio sample for liveness and synthetic voice features.

        Args:
            waveform_bytes: Raw mono wave audio bytes (16kHz)
            claimed_phrase: Optional expected challenge phrase to verify

        Returns:
            SpoofResult indicating if the sample is genuine or spoofed
        """
