"""
app/preprocessing/audio_pipeline.py

Master audio processing pipeline.

Orchestrates the complete preprocessing chain:
  1. Load & normalize audio
  2. Voice Activity Detection (VAD)
  3. Noise suppression (Wiener filter)
  4. Silence trimming
  5. Quality validation
  6. Return validated waveform + quality report

This is the single entry point for all audio processing.
Used by both enrollment and authentication pipelines.

Design: Pipeline returns a ProcessedAudio object with the waveform tensor
ready for ECAPA-TDNN embedding generation, plus complete quality metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.preprocessing.audio_normalizer import AudioNormalizer, get_normalizer
from app.preprocessing.noise_suppressor import NoiseSuppressor, get_noise_suppressor
from app.preprocessing.quality_validator import AudioQualityValidator, QualityReport, get_quality_validator
from app.preprocessing.vad import VADResult, VoiceActivityDetector, get_vad

logger = logging.getLogger(__name__)


@dataclass
class ProcessedAudio:
    """Result of the complete audio preprocessing pipeline."""

    waveform: np.ndarray         # Float32, mono, 16kHz, ready for embedding
    sample_rate: int
    quality: QualityReport
    vad_result: VADResult
    original_duration_seconds: float
    processed_duration_seconds: float
    accepted: bool
    rejection_reason: str | None

    @property
    def waveform_tensor(self):
        """Return waveform as a PyTorch tensor (2D: [1, time])."""
        import torch
        return torch.from_numpy(self.waveform).unsqueeze(0)


class AudioPipeline:
    """
    Full audio preprocessing pipeline.

    Args:
        normalizer: AudioNormalizer instance
        vad: VoiceActivityDetector instance
        suppressor: NoiseSuppressor instance
        validator: AudioQualityValidator instance
    """

    def __init__(
        self,
        normalizer: AudioNormalizer | None = None,
        vad: VoiceActivityDetector | None = None,
        suppressor: NoiseSuppressor | None = None,
        validator: AudioQualityValidator | None = None,
    ) -> None:
        self.normalizer = normalizer or get_normalizer()
        self.vad = vad or get_vad()
        self.suppressor = suppressor or get_noise_suppressor()
        self.validator = validator or get_quality_validator()

    def process_file(
        self,
        path: str | Path,
        for_enrollment: bool = False,
    ) -> ProcessedAudio:
        """
        Process an audio file through the complete pipeline.

        Args:
            path: Path to audio file
            for_enrollment: Apply stricter quality requirements

        Returns:
            ProcessedAudio with validated waveform and quality metadata
        """
        logger.debug("Pipeline: processing file %s (enrollment=%s)", path, for_enrollment)

        # Step 1: Load and normalize
        try:
            audio = self.normalizer.load_and_normalize(path)
        except ValueError as e:
            return self._rejected(str(e))

        return self._run_pipeline(audio, for_enrollment)

    def process_bytes(
        self,
        audio_bytes: bytes,
        for_enrollment: bool = False,
    ) -> ProcessedAudio:
        """
        Process raw audio bytes through the complete pipeline.

        Args:
            audio_bytes: Raw WAV/OGG/FLAC bytes
            for_enrollment: Apply stricter quality requirements

        Returns:
            ProcessedAudio with validated waveform and quality metadata
        """
        try:
            audio = self.normalizer.load_from_bytes(audio_bytes)
        except ValueError as e:
            return self._rejected(str(e))

        return self._run_pipeline(audio, for_enrollment)

    def _run_pipeline(self, audio: np.ndarray, for_enrollment: bool) -> ProcessedAudio:
        """Internal: run stages 2-5 on already-normalized audio."""
        original_duration = len(audio) / self.normalizer.target_sample_rate

        # Step 2: Voice Activity Detection
        vad_result = self.vad.detect(audio)
        if not vad_result.is_speech_sufficient:
            logger.warning("Pipeline VAD rejected audio: %s", vad_result.reason)
            quality = self.validator.validate(audio, vad_result.voice_ratio, for_enrollment)
            return ProcessedAudio(
                waveform=audio,
                sample_rate=self.normalizer.target_sample_rate,
                quality=quality,
                vad_result=vad_result,
                original_duration_seconds=original_duration,
                processed_duration_seconds=original_duration,
                accepted=False,
                rejection_reason=vad_result.reason,
            )

        # Use trimmed audio from VAD
        trimmed = vad_result.trimmed_audio if vad_result.trimmed_audio is not None else audio

        # Step 3: Noise suppression
        denoised = self.suppressor.suppress(trimmed)

        # Step 4: Quality validation
        quality = self.validator.validate(denoised, vad_result.voice_ratio, for_enrollment)
        processed_duration = len(denoised) / self.normalizer.target_sample_rate

        if not quality.accepted:
            logger.warning(
                "Pipeline quality validation rejected audio: %s",
                "; ".join(quality.rejection_reasons),
            )
            return ProcessedAudio(
                waveform=denoised,
                sample_rate=self.normalizer.target_sample_rate,
                quality=quality,
                vad_result=vad_result,
                original_duration_seconds=original_duration,
                processed_duration_seconds=processed_duration,
                accepted=False,
                rejection_reason="; ".join(quality.rejection_reasons),
            )

        logger.info(
            "Pipeline: accepted audio quality=%s snr=%.1fdB voice=%.2f dur=%.2fs",
            quality.quality_level, quality.snr_db, quality.voice_ratio, processed_duration,
        )

        return ProcessedAudio(
            waveform=denoised,
            sample_rate=self.normalizer.target_sample_rate,
            quality=quality,
            vad_result=vad_result,
            original_duration_seconds=original_duration,
            processed_duration_seconds=processed_duration,
            accepted=True,
            rejection_reason=None,
        )

    @staticmethod
    def _rejected(reason: str) -> ProcessedAudio:
        """Return a uniformly rejected ProcessedAudio."""
        from app.preprocessing.quality_validator import QualityLevel, QualityReport
        from app.preprocessing.vad import VADResult
        return ProcessedAudio(
            waveform=np.zeros(1, dtype=np.float32),
            sample_rate=16000,
            quality=QualityReport(
                accepted=False,
                quality_level=QualityLevel.REJECTED,
                quality_score=0.0,
                snr_db=0.0,
                voice_ratio=0.0,
                duration_seconds=0.0,
                peak_amplitude=0.0,
                is_clipped=False,
                dc_offset=0.0,
                rejection_reasons=[reason],
            ),
            vad_result=VADResult(
                is_speech_sufficient=False,
                voice_ratio=0.0,
                total_frames=0,
                voiced_frames=0,
                speech_duration_seconds=0.0,
                total_duration_seconds=0.0,
                trimmed_audio=None,
                reason=reason,
            ),
            original_duration_seconds=0.0,
            processed_duration_seconds=0.0,
            accepted=False,
            rejection_reason=reason,
        )


# Module-level singleton
_pipeline: AudioPipeline | None = None


def get_audio_pipeline() -> AudioPipeline:
    """Return singleton AudioPipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AudioPipeline()
    return _pipeline
