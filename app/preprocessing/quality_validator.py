"""
app/preprocessing/quality_validator.py

Audio quality validation and SNR estimation.

Validates audio samples before they enter the enrollment or authentication
pipeline. Rejects audio that is too noisy, too short/long, too silent,
or otherwise unsuitable for reliable voice biometric extraction.

Validation checks:
  1. Duration: [AUDIO_MIN_DURATION_SECONDS, AUDIO_MAX_DURATION_SECONDS]
  2. SNR:       > AUDIO_MIN_SNR_DB (enrollment: AUDIO_ENROLLMENT_MIN_SNR_DB)
  3. Voice:     VAD voice_ratio > AUDIO_MIN_VOICE_RATIO
  4. Clipping:  Peak amplitude < 0.99 (not clipped)
  5. DC offset: |mean| < 0.01 (no DC bias)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    EXCELLENT = "EXCELLENT"   # SNR > 30dB, voice_ratio > 0.7
    HIGH = "HIGH"             # SNR > 25dB, voice_ratio > 0.5
    MEDIUM = "MEDIUM"         # SNR > 15dB, voice_ratio > 0.35
    LOW = "LOW"               # Barely passes minimum threshold
    REJECTED = "REJECTED"     # Below minimum threshold


@dataclass
class QualityReport:
    """Audio quality assessment result."""
    accepted: bool
    quality_level: QualityLevel
    quality_score: float        # 0.0 → 1.0 overall score
    snr_db: float
    voice_ratio: float
    duration_seconds: float
    peak_amplitude: float
    is_clipped: bool
    dc_offset: float
    rejection_reasons: list[str]

    @property
    def is_enrollment_grade(self) -> bool:
        """True if quality meets stricter enrollment requirements."""
        return self.quality_level in (QualityLevel.HIGH, QualityLevel.EXCELLENT)


class AudioQualityValidator:
    """
    Validates audio quality for voice biometric processing.

    Args:
        sample_rate: Expected sample rate in Hz
        min_duration: Minimum speech duration in seconds
        max_duration: Maximum speech duration in seconds
        min_snr_db: Minimum SNR for standard acceptance (authentication)
        enrollment_min_snr_db: Minimum SNR for enrollment (stricter)
        min_voice_ratio: Minimum fraction of voiced frames
        mode: "auth" or "enrollment" (controls SNR threshold selection)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_duration: float = 1.5,
        max_duration: float = 10.0,
        min_snr_db: float = 10.0,
        enrollment_min_snr_db: float = 20.0,
        min_voice_ratio: float = 0.4,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_snr_db = min_snr_db
        self.enrollment_min_snr_db = enrollment_min_snr_db
        self.min_voice_ratio = min_voice_ratio

    def validate(
        self,
        audio: np.ndarray,
        voice_ratio: float,
        for_enrollment: bool = False,
    ) -> QualityReport:
        """
        Validate an audio sample.

        Args:
            audio: Float32 numpy array, mono, at self.sample_rate
            voice_ratio: Voice ratio from VAD (saves re-computation)
            for_enrollment: If True, applies stricter SNR threshold

        Returns:
            QualityReport with pass/fail and detailed metrics
        """
        rejection_reasons: list[str] = []
        duration = len(audio) / self.sample_rate

        # --- Duration check ---
        if duration < self.min_duration:
            rejection_reasons.append(
                f"Too short: {duration:.2f}s < {self.min_duration}s"
            )
        elif duration > self.max_duration:
            rejection_reasons.append(
                f"Too long: {duration:.2f}s > {self.max_duration}s"
            )

        # --- Clipping check ---
        peak = float(np.max(np.abs(audio)))
        is_clipped = peak > 0.99
        if is_clipped:
            rejection_reasons.append(f"Audio clipped: peak={peak:.3f}")

        # --- DC offset check ---
        dc_offset = float(np.mean(audio))
        if abs(dc_offset) > 0.01:
            rejection_reasons.append(f"DC offset too high: {dc_offset:.4f}")

        # --- SNR estimation ---
        snr_db = self._estimate_snr(audio, voice_ratio)
        snr_threshold = self.enrollment_min_snr_db if for_enrollment else self.min_snr_db
        if snr_db < snr_threshold:
            rejection_reasons.append(
                f"SNR too low: {snr_db:.1f}dB < {snr_threshold:.1f}dB"
            )

        # --- Voice ratio check ---
        if voice_ratio < self.min_voice_ratio:
            rejection_reasons.append(
                f"Voice ratio too low: {voice_ratio:.2%} < {self.min_voice_ratio:.2%}"
            )

        accepted = len(rejection_reasons) == 0
        quality_score = self._compute_quality_score(snr_db, voice_ratio, duration, is_clipped)
        quality_level = self._classify_quality(snr_db, voice_ratio, accepted)

        logger.debug(
            "Quality validation: accepted=%s, snr=%.1fdB, voice=%.2f, "
            "duration=%.2fs, quality=%s",
            accepted, snr_db, voice_ratio, duration, quality_level,
        )

        return QualityReport(
            accepted=accepted,
            quality_level=quality_level,
            quality_score=quality_score,
            snr_db=snr_db,
            voice_ratio=voice_ratio,
            duration_seconds=duration,
            peak_amplitude=peak,
            is_clipped=is_clipped,
            dc_offset=dc_offset,
            rejection_reasons=rejection_reasons,
        )

    def _estimate_snr(self, audio: np.ndarray, voice_ratio: float) -> float:
        """
        Estimate SNR using signal/noise power separation.

        Approximate approach: sort frame energies, use bottom 10% as noise estimate,
        top 50% (voiced) as signal estimate.
        """
        if len(audio) == 0:
            return 0.0

        frame_size = int(self.sample_rate * 0.03)  # 30ms frames
        frames = [audio[i: i + frame_size] for i in range(0, len(audio) - frame_size, frame_size)]
        if not frames:
            return 0.0

        frame_powers = np.array([np.mean(f ** 2) + 1e-10 for f in frames])
        frame_powers_sorted = np.sort(frame_powers)

        noise_power = float(np.mean(frame_powers_sorted[:max(1, len(frame_powers_sorted) // 10)]))
        signal_power = float(np.mean(frame_powers_sorted[len(frame_powers_sorted) // 2:]))

        if noise_power <= 0 or signal_power <= 0:
            return 0.0

        snr = 10 * np.log10(signal_power / noise_power)
        return float(np.clip(snr, -10, 60))

    def _compute_quality_score(
        self,
        snr_db: float,
        voice_ratio: float,
        duration: float,
        is_clipped: bool,
    ) -> float:
        """Compute 0.0-1.0 quality score."""
        if is_clipped:
            return 0.0

        # SNR contribution (0-1): 0dB=0, 40dB=1
        snr_score = float(np.clip(snr_db / 40.0, 0, 1))
        # Voice ratio contribution (0-1)
        vr_score = float(np.clip(voice_ratio / 0.8, 0, 1))
        # Duration contribution: penalize very short or very long
        dur_score = 1.0 if 2.0 <= duration <= 6.0 else 0.7

        return round(0.5 * snr_score + 0.35 * vr_score + 0.15 * dur_score, 3)

    def _classify_quality(
        self, snr_db: float, voice_ratio: float, accepted: bool
    ) -> QualityLevel:
        if not accepted:
            return QualityLevel.REJECTED
        if snr_db > 30 and voice_ratio > 0.7:
            return QualityLevel.EXCELLENT
        if snr_db > 25 and voice_ratio > 0.5:
            return QualityLevel.HIGH
        if snr_db > 15 and voice_ratio > 0.35:
            return QualityLevel.MEDIUM
        return QualityLevel.LOW


# Module-level singleton
_validator: AudioQualityValidator | None = None


def get_quality_validator() -> AudioQualityValidator:
    """Return singleton AudioQualityValidator."""
    global _validator
    if _validator is None:
        _validator = AudioQualityValidator()
    return _validator
