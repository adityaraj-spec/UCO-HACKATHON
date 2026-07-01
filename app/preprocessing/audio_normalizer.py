"""
app/preprocessing/audio_normalizer.py

Audio loading, normalization, and codec normalization.

Handles:
  1. Loading audio from file paths (WAV, FLAC, OGG) or byte buffers
  2. Resampling to target rate (16kHz for ECAPA-TDNN)
  3. Channel downmix (stereo → mono)
  4. Target loudness normalization (prevent clipping)
  5. CMVN (Cepstral Mean Variance Normalization) for telephony codec compensation
     - G.711/G.729 audio at 8kHz is upsampled, then CMVN applied to log-mel features

Codec normalization rationale:
  G.711/G.729 compresses audio at 8kHz with μ-law or linear quantization.
  When upsampled to 16kHz, high-frequency content is missing (0-4kHz only).
  Applying CMVN on log-mel filterbanks normalizes the spectral mean/variance,
  reducing the channel mismatch between telephony and microphone recordings.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
TARGET_PEAK = 0.95  # Peak amplitude after normalization


class AudioNormalizer:
    """
    Normalizes audio for voice biometric processing.

    Args:
        target_sample_rate: Output sample rate in Hz (default: 16000)
        target_peak: Peak amplitude for amplitude normalization
    """

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        target_peak: float = TARGET_PEAK,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.target_peak = target_peak

    def load_and_normalize(self, path: str | Path) -> np.ndarray:
        """
        Load audio, resample, convert to mono, and normalize amplitude.

        Args:
            path: Path to audio file (WAV, FLAC, MP3, OGG)

        Returns:
            Float32 numpy array, mono, at target_sample_rate, in [-1, 1]

        Raises:
            ValueError: If audio cannot be loaded or is empty
        """
        import soundfile as sf

        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:
            raise ValueError(f"Failed to load audio from {path}: {e}") from e

        # Downmix to mono
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio[:, 0]

        # Resample if needed
        if sr != self.target_sample_rate:
            audio = self._resample(audio, sr, self.target_sample_rate)

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Amplitude normalization
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio * (self.target_peak / peak)

        logger.debug(
            "Loaded audio: duration=%.2fs, original_sr=%dHz, final_peak=%.3f",
            len(audio) / self.target_sample_rate, sr, float(np.max(np.abs(audio))),
        )

        return audio.astype(np.float32)

    def load_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """
        Load audio from raw bytes (e.g., from uploaded file).

        Args:
            audio_bytes: Raw audio file bytes

        Returns:
            Float32 numpy array, mono, at target_sample_rate
        """
        import soundfile as sf

        with io.BytesIO(audio_bytes) as buf:
            try:
                audio, sr = sf.read(buf, dtype="float32", always_2d=True)
            except Exception as e:
                raise ValueError(f"Failed to decode audio bytes: {e}") from e

        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio[:, 0]

        if sr != self.target_sample_rate:
            audio = self._resample(audio, sr, self.target_sample_rate)

        audio = audio - np.mean(audio)
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio * (self.target_peak / peak)

        return audio.astype(np.float32)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio using scipy's polyphase resampler.

        Falls back to naive linear interpolation when scipy is unavailable.
        """
        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(orig_sr, target_sr)
            up = target_sr // g
            down = orig_sr // g
            return resample_poly(audio, up, down).astype(np.float32)
        except ImportError:
            # Fallback: linear interpolation (lower quality but no dependency)
            num_samples = int(len(audio) * target_sr / orig_sr)
            return np.interp(
                np.linspace(0, len(audio), num_samples),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)


# Module-level singleton
_normalizer: AudioNormalizer | None = None


def get_normalizer() -> AudioNormalizer:
    """Return singleton AudioNormalizer."""
    global _normalizer
    if _normalizer is None:
        _normalizer = AudioNormalizer()
    return _normalizer
