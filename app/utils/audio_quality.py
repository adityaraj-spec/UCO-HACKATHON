"""
app/utils/audio_quality.py

Audio preprocessing pipeline:
- Voice Activity Detection (VAD) & Speaker Diarisation (dominant speaker segment isolation).
- 2-second minimum duration check applied strictly to diarised dominant speaker speech.
- Perceptual audio hashing for replay protection.
"""

import hashlib
import os
import numpy as np
import torch
from fastapi import HTTPException, status
from app.core.config import get_settings

from app.services.diarisation_service import get_diarisation_service

settings = get_settings()


def diarise_and_extract_dominant_speaker(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    enrolled_embedding: np.ndarray | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Delegate neural speaker diarisation to process-wide DiarisationService singleton.
    Strips silence, segments speech with neural embeddings, matches enrolled speaker or dominant speaker,
    and applies 2-second minimum speech duration check.
    """
    diarisation_svc = get_diarisation_service()
    return diarisation_svc.diarise_speech(
        waveform, sample_rate=sample_rate, enrolled_embedding=enrolled_embedding
    )


def compute_perceptual_audio_hash(waveform: torch.Tensor, sample_rate: int = 16000) -> str:
    """
    Compute a perceptual audio fingerprint hash based on spectral energy across frequency bands.
    Used for 24-hour replay attack detection.
    """
    audio_np = waveform.squeeze().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)

    fft_mag = np.abs(np.fft.rfft(audio_np[: min(len(audio_np), sample_rate * 3)]))
    if len(fft_mag) > 64:
        bin_size = len(fft_mag) // 64
        binned = np.array([np.mean(fft_mag[i * bin_size : (i + 1) * bin_size]) for i in range(64)])
    else:
        binned = fft_mag

    binned_str = np.array2string(np.round(binned, 2), separator=",")
    return hashlib.sha256(binned_str.encode("utf-8")).hexdigest()


def estimate_snr_db(waveform: torch.Tensor) -> float:
    """
    Estimate signal-to-noise ratio from frame RMS energy.

    This is a lightweight quality gate, not a laboratory SNR measurement:
    the 90th percentile RMS represents active speech and the 10th percentile
    represents the noise floor.
    """
    audio_np = waveform.squeeze().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)
    if audio_np.size == 0:
        return 0.0

    frame_len = max(1, int(settings.TARGET_SAMPLE_RATE * 0.025))
    hop_len = max(1, int(settings.TARGET_SAMPLE_RATE * 0.010))
    rms = []
    for start in range(0, max(1, len(audio_np) - frame_len + 1), hop_len):
        frame = audio_np[start : start + frame_len]
        rms.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
    if not rms:
        return 0.0

    rms_arr = np.asarray(rms, dtype=np.float64)
    speech_rms = float(np.percentile(rms_arr, 90))
    noise_rms = float(np.percentile(rms_arr, 10))
    return float(20.0 * np.log10(max(speech_rms, 1e-9) / max(noise_rms, 1e-9)))


def validate_enrollment_audio_quality(waveform: torch.Tensor, sample_rate: int) -> tuple[float, float]:
    """
    Validate enrollment audio duration and estimated SNR.

    Returns:
        (duration_seconds, estimated_snr_db)
    """
    duration_sec = float(waveform.shape[-1] / sample_rate)
    if duration_sec < settings.MIN_SPEECH_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Insufficient enrollment audio duration: {duration_sec:.2f}s "
                f"(minimum required: {settings.MIN_SPEECH_DURATION_SEC:.1f}s)"
            ),
        )

    min_snr_db = float(os.environ.get("ENROLLMENT_MIN_SNR_DB", "10.0"))
    snr_db = estimate_snr_db(waveform)
    if snr_db < min_snr_db:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Enrollment audio SNR is too low: {snr_db:.2f} dB "
                f"(minimum required: {min_snr_db:.1f} dB)"
            ),
        )
    return duration_sec, snr_db
