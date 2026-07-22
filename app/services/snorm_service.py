"""
app/services/snorm_service.py

Adaptive Score Normalisation (s-norm) service using a dynamic real impostor speaker cohort.
"""

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.impostor_cohort import ImpostorCohort
from app.models.voiceprint import Voiceprint
from app.services.biohash_service import BioHashService

settings = get_settings()

MIN_REAL_SNORM_COHORT_SIZE = 5
SYNTHETIC_COHORT_SEED = 20260722


class SNormService:
    """Manages impostor cohort score normalisation for speaker verification."""

    def __init__(self, biohash_service: BioHashService | None = None):
        self.biohash_service = biohash_service or BioHashService()

    async def compute_snorm_score(
        self,
        session: AsyncSession,
        raw_similarity: float,
        live_biohash: str,
    ) -> tuple[float, float, float]:
        """
        Compute s-norm normalised score: s_norm = (raw_similarity - mu_imp) / sigma_imp.
        Returns tuple: (snorm_score, mu_imp, sigma_imp).
        """
        # Fetch real impostor cohort embeddings.
        result = await session.execute(
            select(ImpostorCohort.embedding).limit(settings.IMPOSTOR_COHORT_SIZE)
        )
        cohort_embeddings = list(result.scalars().all())

        if len(cohort_embeddings) < MIN_REAL_SNORM_COHORT_SIZE:
            synthetic_count = MIN_REAL_SNORM_COHORT_SIZE - len(cohort_embeddings)
            cohort_embeddings.extend(
                self._synthetic_impostor_embeddings(count=synthetic_count)
            )

        # Compute similarities between live vector and impostor cohort
        impostor_sims = []
        for imp_emb in cohort_embeddings:
            imp_biohash = self.biohash_service.compute_biohash(imp_emb)
            sim = self.biohash_service.biohash_similarity(live_biohash, imp_biohash)
            impostor_sims.append(sim)

        mu_imp = float(np.mean(impostor_sims))
        sigma_imp = float(np.std(impostor_sims)) + 1e-6  # Epsilon to prevent division by zero

        snorm_score = (raw_similarity - mu_imp) / sigma_imp
        return snorm_score, mu_imp, sigma_imp

    def _synthetic_impostor_embeddings(self, count: int) -> list[list[float]]:
        """
        Build deterministic random unit vectors for tiny demo/dev databases.

        These vectors are not calibrated from real speaker distributions; they
        are used only to bridge below the minimum real cohort size.
        """
        if count <= 0:
            return []

        rng = np.random.default_rng(SYNTHETIC_COHORT_SEED)
        embeddings = rng.normal(size=(count, settings.EMBEDDING_DIM)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-6)
        return embeddings.astype(float).tolist()

    def evaluate_decision(self, snorm_score: float) -> tuple[str, bool]:
        """
        Evaluate identity decision based on normalised score.
        Returns (decision, verified_bool).
        """
        if snorm_score >= settings.SNORM_PASS_THRESHOLD:
            return "verified", True
        elif snorm_score >= settings.SNORM_STEP_UP_THRESHOLD:
            return "step_up", False
        else:
            return "mismatch", False

    async def refresh_impostor_cohort(self, session: AsyncSession) -> int:
        """
        Periodically populate/refresh impostor cohort with non-matching voiceprint embeddings.
        """
        result = await session.execute(
            select(Voiceprint.embedding).order_by(func.random()).limit(settings.IMPOSTOR_COHORT_SIZE)
        )
        voiceprints = result.scalars().all()

        if not voiceprints:
            return 0

        # Clear existing cohort
        await session.execute(select(ImpostorCohort))
        for vp_emb in voiceprints:
            session.add(ImpostorCohort(embedding=list(vp_emb)))
        
        await session.commit()
        return len(voiceprints)
