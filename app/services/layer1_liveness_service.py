"""
app/services/layer1_liveness_service.py

Thin service wrapper around the existing PhaseGuard Layer 1 audio
authenticity model. Verification currently accepts a caller-provided
layer1_score; enrollment uses this service directly so synthetic audio
cannot become a durable voiceprint anchor.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.utils.exceptions import EnrollmentRejectedError

settings = get_settings()
log = get_logger(__name__)


class Layer1LivenessService:
    """Runs the existing Layer 1 model and returns AI-voice probability."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path or os.environ.get("LAYER1_MODEL_PATH", "models/layer1_mobilenet.pth"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def predict_ai_probability(self, audio_path: str) -> float:
        """Return Layer 1 AI/deepfake probability in [0, 1]."""
        model = self._load_model()
        try:
            from scripts.features import extract_all

            result = extract_all(audio_path, sr=settings.TARGET_SAMPLE_RATE)
            if result is None:
                raise EnrollmentRejectedError("Layer 1 could not extract audio features.")

            mel_t = torch.FloatTensor(result["mel"]).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return float(model(mel_t)[0][0])
        except EnrollmentRejectedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EnrollmentRejectedError(f"Layer 1 check failed: {exc}") from exc

    def _load_model(self):
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise EnrollmentRejectedError(
                f"Layer 1 model is unavailable at {self.model_path}; refusing enrollment."
            )

        from scripts.train import PhaseGuardL1

        model = PhaseGuardL1().to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        self._model = model
        log.info("Loaded Layer 1 liveness model from %s", self.model_path)
        return self._model
