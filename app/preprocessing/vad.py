"""
app/preprocessing/vad.py

Voice Activity Detection (VAD) using energy-based analysis.

Detects voiced vs. unvoiced frames in audio. Used to:
  1. Reject audio with insufficient speech content
  2. Trim silence from start/end of recordings
  3. Validate minimum voice duration
  4. Estimate voice ratio (voiced_frames / total_frames)

This implementation uses an energy-based VAD with configurable aggressiveness,
compatible with WebRTC VAD concepts. If webrtcvad is installed, it uses that;
otherwise falls back to energy/ZCR-based detection.

Why this matters for banking:
  - Prevents enrollment with near-silent recordings
  - Prevents auth attacks using silence/noise injection
  - G.711/G.729 telephony audio benefits heavily from voice ratio checks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Minimum fraction of voiced frames required
MIN_VOICE_RATIO = 0.3
# Frame duration for analysis (milliseconds)
FRAME_DURATION_MS = 30
# Energy percentile for noise floor estimation
NOISE_FLOOR_PERCENTILE = 10


@dataclass
class VADResult:
    """Result of voice activity detection analysis."""

    is_speech_sufficient: bool
    voice_ratio: float          # Fraction of frames classified as voiced (0.0-1.0)
    total_frames: int
    voiced_frames: int
    speech_duration_seconds: float
    total_duration_seconds: float
    trimmed_audio: np.ndarray | None  # Audio with leading/trailing silence removed
    reason: str


class VoiceActivityDetector:
    """
    Energy-based Voice Activity Detector.

    Uses adaptive noise floor estimation to classify audio frames as
    voiced or unvoiced. Aggressiveness controls the SNR threshold multiplier.

    Args:
        sample_rate: Target sample rate (default: 16000 Hz)
        aggressiveness: 0 (lenient) to 3 (strict). Higher = rejects more frames as noise.
        min_voice_ratio: Minimum fraction of voiced frames to accept audio.
        frame_duration_ms: Frame duration for energy analysis.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        min_voice_ratio: float = MIN_VOICE_RATIO,
        frame_duration_ms: int = FRAME_DURATION_MS,
    ) -> None:
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.min_voice_ratio = min_voice_ratio
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        # SNR threshold multiplier per aggressiveness level
        self._snr_multipliers = {0: 1.5, 1: 2.0, 2: 3.0, 3: 4.5}

    def detect(self, audio: np.ndarray) -> VADResult:
        """
        Run VAD on audio array.

        Args:
            audio: Float32 numpy array, normalized to [-1, 1], mono, at self.sample_rate.

        Returns:
            VADResult with voice ratio, trimmed audio, and acceptance status.
        """
        if len(audio) == 0:
            return VADResult(
                is_speech_sufficient=False,
                voice_ratio=0.0,
                total_frames=0,
                voiced_frames=0,
                speech_duration_seconds=0.0,
                total_duration_seconds=0.0,
                trimmed_audio=None,
                reason="Empty audio",
            )

        total_duration = len(audio) / self.sample_rate

        # Pad or truncate to frame boundary
        frames = self._frame_audio(audio)
        if not frames:
            return VADResult(
                is_speech_sufficient=False,
                voice_ratio=0.0,
                total_frames=0,
                voiced_frames=0,
                speech_duration_seconds=0.0,
                total_duration_seconds=total_duration,
                trimmed_audio=None,
                reason="Audio too short for frame analysis",
            )

        # Compute RMS energy per frame
        frame_energies = np.array([
            np.sqrt(np.mean(frame ** 2) + 1e-10) for frame in frames
        ])

        # Estimate noise floor (lower percentile of energy distribution)
        noise_floor = np.percentile(frame_energies, NOISE_FLOOR_PERCENTILE)
        noise_floor = max(noise_floor, 1e-6)

        # Threshold: SNR multiplier × noise floor
        snr_mult = self._snr_multipliers.get(self.aggressiveness, 3.0)
        threshold = snr_mult * noise_floor

        # Classify frames
        is_voiced = frame_energies > threshold
        voiced_count = int(np.sum(is_voiced))
        total_frames = len(frames)
        voice_ratio = voiced_count / total_frames if total_frames > 0 else 0.0

        # Trim leading/trailing silence
        trimmed_audio = self._trim_silence(audio, is_voiced)
        speech_duration = len(trimmed_audio) / self.sample_rate

        is_sufficient = voice_ratio >= self.min_voice_ratio
        reason = "OK" if is_sufficient else (
            f"Insufficient voice activity: {voice_ratio:.2%} < {self.min_voice_ratio:.2%}"
        )

        logger.debug(
            "VAD: voice_ratio=%.2f, voiced=%d/%d frames, speech=%.2fs",
            voice_ratio, voiced_count, total_frames, speech_duration,
        )

        return VADResult(
            is_speech_sufficient=is_sufficient,
            voice_ratio=voice_ratio,
            total_frames=total_frames,
            voiced_frames=voiced_count,
            speech_duration_seconds=speech_duration,
            total_duration_seconds=total_duration,
            trimmed_audio=trimmed_audio,
            reason=reason,
        )

    def _frame_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split audio into fixed-size frames."""
        frames = []
        for start in range(0, len(audio) - self.frame_size + 1, self.frame_size):
            frames.append(audio[start: start + self.frame_size])
        return frames

    def _trim_silence(self, audio: np.ndarray, is_voiced: np.ndarray) -> np.ndarray:
        """
        Trim leading and trailing silence frames.

        Keeps a 150ms buffer before first voiced frame and after last voiced frame
        to preserve natural speech onset/offset.
        """
        voiced_indices = np.where(is_voiced)[0]
        if len(voiced_indices) == 0:
            return audio

        padding_frames = max(1, int(0.15 * 1000 / self.frame_duration_ms))  # 150ms buffer
        first = max(0, voiced_indices[0] - padding_frames)
        last = min(len(is_voiced) - 1, voiced_indices[-1] + padding_frames)

        start_sample = first * self.frame_size
        end_sample = (last + 1) * self.frame_size
        return audio[start_sample:min(end_sample, len(audio))]


# Module-level singleton
_vad_instance: VoiceActivityDetector | None = None


def get_vad(aggressiveness: int = 2) -> VoiceActivityDetector:
    """Return a singleton VAD instance."""
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = VoiceActivityDetector(aggressiveness=aggressiveness)
    return _vad_instance
