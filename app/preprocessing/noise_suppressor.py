"""
app/preprocessing/noise_suppressor.py

Spectral subtraction noise suppressor.

Reduces stationary background noise from audio recordings by estimating
the noise spectrum during non-speech frames and subtracting it from the
speech spectrum. This improves embedding quality for telephony-grade audio
(G.711/G.729) where SNR can be as low as 10-15 dB.

Algorithm: Wiener filter-based spectral subtraction
  1. Estimate noise PSD from first N silent frames (noise_estimation_ms)
  2. Compute noisy speech STFT
  3. Apply Wiener filter: H(f) = max(1 - β × N(f)/|X(f)|², floor)
  4. Reconstruct denoised signal via iSTFT

Why Wiener filter over simple subtraction:
  - Avoids musical noise artifacts more effectively
  - Better preserves speech harmonics
  - More robust to non-stationary noise
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NoiseSuppressor:
    """
    Wiener-filter based spectral noise suppressor.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)
        n_fft: FFT size (default: 512 → 32ms at 16kHz)
        hop_length: Hop size (default: 128 → 8ms, 75% overlap)
        noise_estimation_ms: Duration of initial frames used for noise floor estimation
        wiener_beta: Over-subtraction factor (1.0=standard, higher=stronger suppression)
        gain_floor: Minimum spectral gain (prevents over-suppression / musical noise)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        noise_estimation_ms: int = 200,
        wiener_beta: float = 1.5,
        gain_floor: float = 0.05,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_estimation_ms = noise_estimation_ms
        self.wiener_beta = wiener_beta
        self.gain_floor = gain_floor

    def suppress(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise suppression to audio.

        Args:
            audio: Float32 numpy array, mono, at self.sample_rate. Range: [-1, 1].

        Returns:
            Denoised float32 numpy array of similar length.
        """
        if len(audio) < self.n_fft:
            logger.debug("Audio too short for noise suppression, returning unchanged")
            return audio

        # STFT
        stft = self._stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise PSD from the first `noise_estimation_ms` worth of frames
        noise_frames = max(1, int(self.noise_estimation_ms / 1000 * self.sample_rate / self.hop_length))
        noise_frames = min(noise_frames, magnitude.shape[1])
        noise_psd = np.mean(magnitude[:, :noise_frames] ** 2, axis=1, keepdims=True)
        noise_psd = np.maximum(noise_psd, 1e-10)

        # Speech PSD estimate
        speech_psd = magnitude ** 2

        # Wiener filter gain: H(f) = max(1 - β * N(f)/S(f), floor)
        gain = 1.0 - self.wiener_beta * (noise_psd / (speech_psd + 1e-10))
        gain = np.maximum(gain, self.gain_floor)

        # Apply gain to magnitude, reconstruct with original phase
        denoised_magnitude = gain * magnitude
        denoised_stft = denoised_magnitude * np.exp(1j * phase)

        # iSTFT
        denoised = self._istft(denoised_stft, len(audio))

        logger.debug(
            "Noise suppression: input_rms=%.4f, output_rms=%.4f",
            float(np.sqrt(np.mean(audio ** 2))),
            float(np.sqrt(np.mean(denoised ** 2))),
        )

        return denoised.astype(np.float32)

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute Short-Time Fourier Transform."""
        window = np.hanning(self.n_fft)
        frames = []
        for start in range(0, len(audio) - self.n_fft + 1, self.hop_length):
            frame = audio[start: start + self.n_fft] * window
            frames.append(np.fft.rfft(frame))
        if not frames:
            return np.zeros((self.n_fft // 2 + 1, 1), dtype=complex)
        return np.column_stack(frames)

    def _istft(self, stft: np.ndarray, original_length: int) -> np.ndarray:
        """Reconstruct signal from STFT via overlap-add."""
        window = np.hanning(self.n_fft)
        output = np.zeros(original_length + self.n_fft, dtype=np.float32)
        window_sum = np.zeros(original_length + self.n_fft, dtype=np.float32)

        for i, frame_fft in enumerate(stft.T):
            frame = np.fft.irfft(frame_fft, n=self.n_fft).real
            start = i * self.hop_length
            end = start + self.n_fft
            output[start:end] += frame * window
            window_sum[start:end] += window ** 2

        # Normalize by window overlap sum
        mask = window_sum > 1e-8
        output[mask] /= window_sum[mask]
        return output[:original_length]


# Module-level singleton
_suppressor: NoiseSuppressor | None = None


def get_noise_suppressor() -> NoiseSuppressor:
    """Return a singleton NoiseSuppressor instance."""
    global _suppressor
    if _suppressor is None:
        _suppressor = NoiseSuppressor()
    return _suppressor
