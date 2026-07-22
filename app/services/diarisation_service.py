"""
app/services/diarisation_service.py

Neural Speaker Diarisation service using neural ECAPA-TDNN frame-level speaker embeddings
and cosine distance agglomerative clustering.

Pipeline:
1. Run VAD to strip silence.
2. Segment active speech into sliding windows and extract neural speaker embeddings using ECAPA-TDNN.
3. Cluster embeddings to partition speech into distinct speaker identities.
4. If enrolled_embedding is provided (1-to-1 verification), match speaker segments by embedding cosine similarity to the enrolled voiceprint.
5. If enrolled_embedding is None (1-to-N identification), isolate segments of the dominant speaker (longest total speech duration).
6. Enforce 2-second minimum duration check on diarised target speaker speech.
"""

import threading
import numpy as np
import torch
from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.ecapa_service import ECAPAService, get_ecapa_service

settings = get_settings()
log = get_logger(__name__)


class DiarisationService:
    """Neural Speaker Diarisation service loaded once at startup, stateless during inference."""

    def __init__(self, ecapa_service: ECAPAService | None = None):
        self.ecapa_service = ecapa_service or get_ecapa_service()
        self._lock = threading.Lock()

    def diarise_speech(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        enrolled_embedding: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Perform neural speaker diarisation on input audio.

        Args:
            waveform: Raw input audio tensor (1, T) or (T,)
            sample_rate: Audio sampling rate (default 16000 Hz)
            enrolled_embedding: Optional 192-dim enrolled voiceprint vector for 1-to-1 target speaker matching.

        Returns:
            (target_speaker_waveform_tensor, target_speaker_duration_seconds)
        """
        audio_np = waveform.squeeze().cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)

        window_samples = int(sample_rate * 0.5)  # 500ms sliding window
        hop_samples = int(sample_rate * 0.25)    # 250ms hop

        # 1. Run Energy/VAD Silence Stripping
        frame_len = int(sample_rate * 0.025)
        hop_len = int(sample_rate * 0.010)
        energies = np.array([
            np.sum(audio_np[i : i + frame_len] ** 2)
            for i in range(0, max(1, len(audio_np) - frame_len), hop_len)
        ])

        if len(energies) == 0:
            active_audio = audio_np
        else:
            mean_e = np.mean(energies)
            vad_thresh = max(1e-7, mean_e * 0.1)
            active_mask = energies >= vad_thresh

            active_samples = []
            for idx, active in enumerate(active_mask):
                if active:
                    start = idx * hop_len
                    end = start + frame_len
                    active_samples.extend(audio_np[start:end])
            active_audio = np.array(active_samples, dtype=np.float32) if active_samples else audio_np

        if len(active_audio) < window_samples:
            diarised_np = active_audio
            duration = float(len(active_audio) / sample_rate)
        else:
            # 2. Extract Neural Speaker Embeddings per Window
            window_audio_chunks = []
            window_embeddings = []

            for start in range(0, len(active_audio) - window_samples + 1, hop_samples):
                chunk = active_audio[start : start + window_samples]
                window_audio_chunks.append(chunk)

                # Save chunk to temp tensor waveform for ECAPA embedding
                chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
                with torch.no_grad():
                    model = self.ecapa_service.load_model()
                    emb_tensor = model.encode_batch(chunk_tensor)
                    emb = emb_tensor.squeeze().cpu().numpy().flatten()
                    norm = np.linalg.norm(emb)
                    emb_norm = (emb / max(norm, 1e-6)).astype(np.float32)
                    window_embeddings.append(emb_norm)

            window_embeddings_np = np.array(window_embeddings)

            # 3. Neural Speaker Clustering (Cosine Distance Agglomerative Clustering)
            num_windows = len(window_embeddings_np)
            speaker_labels = np.zeros(num_windows, dtype=int)

            if num_windows >= 4:
                # Group windows into clusters based on cosine distance threshold 0.35
                clusters: list[list[int]] = []
                for i in range(num_windows):
                    emb = window_embeddings_np[i]
                    assigned = False
                    for c_idx, cl in enumerate(clusters):
                        centroid = np.mean(window_embeddings_np[cl], axis=0)
                        sim = float(np.dot(emb, centroid) / (np.linalg.norm(centroid) + 1e-6))
                        if sim >= 0.65:
                            cl.append(i)
                            assigned = True
                            break
                    if not assigned:
                        clusters.append([i])

                for c_idx, cl in enumerate(clusters):
                    for w_idx in cl:
                        speaker_labels[w_idx] = c_idx

            unique_speakers = np.unique(speaker_labels)

            # 4. Target Speaker Selection
            if enrolled_embedding is not None and len(unique_speakers) > 1:
                # 1-to-1 Match: Compare each speaker cluster centroid to enrolled voiceprint
                enrolled_norm = np.asarray(enrolled_embedding, dtype=np.float32)
                enrolled_norm = enrolled_norm / (np.linalg.norm(enrolled_norm) + 1e-6)

                best_speaker = unique_speakers[0]
                best_sim = -1.0

                for spk in unique_speakers:
                    spk_indices = np.where(speaker_labels == spk)[0]
                    cluster_centroid = np.mean(window_embeddings_np[spk_indices], axis=0)
                    sim = float(np.dot(cluster_centroid, enrolled_norm))
                    if sim > best_sim:
                        best_sim = sim
                        best_speaker = spk

                target_speaker_id = best_speaker
                log.info(f"Diarisation matched target enrolled speaker {target_speaker_id} with similarity {best_sim:.4f}")
            else:
                # 1-to-N Identification or Single Speaker: Select dominant speaker (longest total speech duration)
                speaker_durations = {
                    spk: np.sum(speaker_labels == spk) * 0.250 for spk in unique_speakers
                }
                target_speaker_id = max(speaker_durations, key=speaker_durations.get)
                log.info(f"Diarisation selected dominant speaker {target_speaker_id}")

            # Collect target speaker sample mask
            target_indices = np.where(speaker_labels == target_speaker_id)[0]
            target_mask = np.zeros(len(active_audio), dtype=bool)

            for idx in target_indices:
                start_samp = idx * hop_samples
                end_samp = min(len(active_audio), start_samp + window_samples)
                target_mask[start_samp:end_samp] = True

            diarised_np = active_audio[target_mask] if np.any(target_mask) else active_audio
            duration = float(len(diarised_np) / sample_rate)

        # 5. Enforce 2-Second Minimum Duration Check on Diarised Target Speaker Speech
        if settings.ENABLE_AUDIO_QUALITY_CHECK and duration < settings.MIN_SPEECH_DURATION_SEC:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Insufficient speech for verification: diarised target speaker speech "
                    f"is {duration:.2f}s (minimum required: {settings.MIN_SPEECH_DURATION_SEC:.1f}s)"
                ),
            )

        diarised_tensor = torch.from_numpy(diarised_np.astype(np.float32)).unsqueeze(0)
        return diarised_tensor, duration


# Singleton instance
_diarisation_service: DiarisationService | None = None
_diarisation_lock = threading.Lock()


def get_diarisation_service() -> DiarisationService:
    """Return process-wide DiarisationService singleton loaded once at startup."""
    global _diarisation_service
    if _diarisation_service is None:
        with _diarisation_lock:
            if _diarisation_service is None:
                _diarisation_service = DiarisationService()
    return _diarisation_service
