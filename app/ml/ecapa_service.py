"""
app/ml/ecapa_service.py

ML service wrapping SpeechBrain's pretrained ECAPA-TDNN speaker recognition
model (speechbrain/spkrec-ecapa-voxceleb).

This module is intentionally framework-agnostic with respect to the web
layer: it has no FastAPI imports and can be exercised directly from tests or
offline scripts.

Public API:
    load_model()                       -> SpeakerRecognition
    extract_embedding(audio_file)      -> np.ndarray, shape (embedding_dim,)
    enroll_user(audio_files)           -> (np.ndarray, int) averaged embedding + count
    verify_user(audio_file, enrolled)  -> float cosine similarity
    cosine_similarity(a, b)            -> float

A module-level singleton (`get_ecapa_service`) ensures the ~80-100MB model is
loaded into memory only once per process, since loading it is the most
expensive operation in the entire pipeline.
"""

from __future__ import annotations

import threading

import numpy as np
import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.utils.audio_utils import load_waveform
from app.utils.exceptions import EmbeddingExtractionError, ModelLoadError

settings = get_settings()
log = get_logger(__name__)


class ECAPAService:
    """
    Thin wrapper around SpeechBrain's pretrained ECAPA-TDNN speaker
    recognition model, providing the operations PhaseGuard's Layer 2 needs:
    embedding extraction, enrollment (averaging), and verification.
    """

    def __init__(
        self,
        model_source: str | None = None,
        save_dir: str | None = None,
        target_sr: int | None = None,
    ) -> None:
        self.model_source = model_source or settings.ECAPA_MODEL_SOURCE
        self.save_dir = save_dir or settings.ECAPA_MODEL_SAVE_DIR
        self.target_sr = target_sr or settings.TARGET_SAMPLE_RATE
        self._model = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        """
        Load (or return the already-loaded) SpeechBrain ECAPA-TDNN model.

        Thread-safe: only one thread will trigger the actual download/load;
        concurrent callers block until it is ready.
        """
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:  # re-check after acquiring lock
                return self._model

            log.info(
                f"Loading ECAPA-TDNN model '{self.model_source}' "
                f"into '{self.save_dir}' (this may take a while on first run)..."
            )
            try:
                # SpeechBrain >= 1.0 moved pretrained interfaces to
                # speechbrain.inference, but kept a `speechbrain.pretrained`
                # alias for backwards compatibility. We try the new location
                # first and fall back gracefully.
                try:
                    from speechbrain.inference.speaker import SpeakerRecognition
                except ImportError:  # pragma: no cover - older speechbrain
                    from speechbrain.pretrained import SpeakerRecognition

                self._model = SpeakerRecognition.from_hparams(
                    source=self.model_source,
                    savedir=self.save_dir,
                    run_opts={"device": "cpu"},
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("Failed to load ECAPA-TDNN model")
                raise ModelLoadError(str(exc)) from exc

            log.info("ECAPA-TDNN model loaded successfully.")
            return self._model

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------
    def extract_embedding(self, audio_file: str) -> np.ndarray:
        """
        Extract a speaker embedding vector from a single audio file.

        Args:
            audio_file: Path to a WAV/FLAC/MP3 file on disk.

        Returns:
            A 1-D numpy array of length `settings.EMBEDDING_DIM`.

        Raises:
            EmbeddingExtractionError: if loading the audio or running the
            model fails.
        """
        model = self.load_model()

        try:
            waveform, _ = load_waveform(audio_file, target_sr=self.target_sr)

            with torch.no_grad():
                embedding_tensor = model.encode_batch(waveform)

            raw_embedding = embedding_tensor.squeeze().detach().cpu().numpy()
            raw_embedding = np.asarray(raw_embedding, dtype=np.float32).flatten()

            raw_norm = float(np.linalg.norm(raw_embedding))

            if raw_embedding.shape[0] != settings.EMBEDDING_DIM:
                log.warning(
                    f"Unexpected embedding dimension {raw_embedding.shape[0]} "
                    f"(expected {settings.EMBEDDING_DIM}) for '{audio_file}'"
                )

            # L2 normalize
            norm_val = np.linalg.norm(raw_embedding)
            normalized_embedding = (raw_embedding / max(norm_val, 1e-6)).astype(np.float32)

            return normalized_embedding
        except EmbeddingExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception(f"Embedding extraction failed for '{audio_file}'")
            raise EmbeddingExtractionError(audio_file, str(exc)) from exc

    def extract_embedding_with_confidence(self, audio_file: str) -> tuple[np.ndarray, float, float]:
        """
        Extract embedding and compute model-derived confidence score from raw L2 norm.

        Returns:
            (normalized_embedding, confidence_score, raw_l2_norm)
        """
        embedding = self.extract_embedding(audio_file)
        raw_norm = float(np.linalg.norm(embedding))
        # Compute confidence score normalized to [0.0, 1.0]
        confidence = float(np.clip(raw_norm / settings.TARGET_NORM_SCALE, 0.1, 1.0))
        return embedding, confidence, raw_norm

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
    def enroll_user(self, audio_files: list[str]) -> tuple[np.ndarray, int]:
        """
        Build a voiceprint by extracting embeddings from multiple
        utterances and averaging them.

        Args:
            audio_files: List of paths to enrollment WAV files
                         (recommended: 30-50 utterances).

        Returns:
            A tuple of:
              - averaged embedding (np.ndarray, shape (embedding_dim,))
              - number of files successfully processed.

        Files that fail to process (corrupt audio etc.) are skipped with a
        warning rather than aborting the entire enrollment, so a few bad
        recordings don't block the whole batch.
        """
        embeddings: list[np.ndarray] = []

        for audio_file in audio_files:
            try:
                embeddings.append(self.extract_embedding(audio_file))
            except EmbeddingExtractionError as exc:
                log.warning(f"Skipping enrollment file due to error: {exc.message}")
                continue

        if not embeddings:
            return np.zeros(settings.EMBEDDING_DIM, dtype=np.float32), 0

        stacked = np.stack(embeddings, axis=0)
        averaged = stacked.mean(axis=0).astype(np.float32)

        log.info(
            f"Enrollment voiceprint computed from {len(embeddings)}/"
            f"{len(audio_files)} recordings."
        )
        return averaged, len(embeddings)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def verify_user(
        self, audio_file: str, enrolled_embedding: np.ndarray | list[float]
    ) -> float:
        """
        Legacy helper that computes cosine similarity between a live audio
        sample and a stored voiceprint.

        Args:
            audio_file: Path to the live/incoming audio file.
            enrolled_embedding: The stored voiceprint embedding for the
                                 claimed user.

        Returns:
            Cosine similarity score in the range [-1.0, 1.0]. The live
            verification endpoint does not use this helper for the returned
            identity decision; it uses BioHash similarity plus s-norm.
        """
        live_embedding = self.extract_embedding(audio_file)
        enrolled = np.asarray(enrolled_embedding, dtype=np.float32)
        return self.cosine_similarity(live_embedding, enrolled)

    # ------------------------------------------------------------------
    # Math helper
    # ------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two vectors.

        cosine_similarity(a, b) = (a . b) / (||a|| * ||b||)

        Returns 0.0 if either vector has zero magnitude (degenerate case),
        to avoid division-by-zero errors.
        """
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))

        # Numerical safety: clamp to valid cosine similarity range.
        return max(-1.0, min(1.0, similarity))


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------
_ecapa_service: ECAPAService | None = None
_singleton_lock = threading.Lock()


def get_ecapa_service() -> ECAPAService:
    """
    Return the process-wide ECAPAService singleton, creating it on first
    use. Use this in FastAPI dependencies / services instead of
    instantiating ECAPAService directly, so the model is loaded only once.
    """
    global _ecapa_service
    if _ecapa_service is None:
        with _singleton_lock:
            if _ecapa_service is None:
                _ecapa_service = ECAPAService()
    return _ecapa_service


# ----------------------------------------------------------------------
# Module-level convenience functions (per the required ML interface)
# ----------------------------------------------------------------------
def load_model():
    """Load (or fetch cached) ECAPA-TDNN model via the singleton service."""
    return get_ecapa_service().load_model()


def extract_embedding(audio_file: str) -> np.ndarray:
    """Extract a speaker embedding from a single audio file."""
    return get_ecapa_service().extract_embedding(audio_file)


def enroll_user(audio_files: list[str]) -> tuple[np.ndarray, int]:
    """Build an averaged voiceprint from a list of enrollment audio files."""
    return get_ecapa_service().enroll_user(audio_files)


def verify_user(audio_file: str, user_id: str, enrolled_embedding) -> float:
    """
    Verify a live audio sample against a stored voiceprint.

    Note: `user_id` is accepted for interface compatibility / logging
    purposes; the actual comparison is performed against
    `enrolled_embedding`, which the caller (service layer) fetches from the
    database for that user.
    """
    score = get_ecapa_service().verify_user(audio_file, enrolled_embedding)
    log.debug(f"Verification score for user_id={user_id}: {score:.4f}")
    return score


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    return ECAPAService.cosine_similarity(a, b)
