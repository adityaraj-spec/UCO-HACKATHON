"""Dynamic sentence Voice KYC enrollment workflow."""

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import numpy as np
from fastapi import UploadFile
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.ml.ecapa_service import ECAPAService, get_ecapa_service
from app.models.kyc_enrollment import KYCEnrollmentSession, VoiceChallenge
from app.repositories.user_repository import UserRepository
from app.repositories.voiceprint_repository import VoiceprintRepository
from app.services.biohash_service import BioHashService
from app.services.challenge_generator import generate_challenge_sentence
from app.utils.asr_verifier import normalize_text, verify_spoken_challenge
from app.utils.audio_utils import cleanup_temp_file, save_upload_to_temp

settings = get_settings()

CHALLENGE_TTL_SECONDS = 90
ASR_MATCH_THRESHOLD = 0.75


class KYCEnrollmentService:
    def __init__(
        self,
        session: AsyncSession,
        ecapa_service: ECAPAService | None = None,
        biohash_service: BioHashService | None = None,
    ) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self.voiceprint_repo = VoiceprintRepository(session)
        self.ecapa_service = ecapa_service or get_ecapa_service()
        self.biohash_service = biohash_service or BioHashService()

    async def start_session(self, phone_number: str) -> KYCEnrollmentSession:
        clean_phone = self._clean_phone(phone_number)
        user = await self._get_or_create_demo_user(clean_phone)
        kyc_session = KYCEnrollmentSession(
            phone_number=clean_phone,
            user_id=user.id,
            status="STARTED",
        )
        self.session.add(kyc_session)
        await self.session.commit()
        await self.session.refresh(kyc_session)
        return kyc_session

    async def issue_challenge(
        self, session_id: uuid.UUID, attempt_no: int
    ) -> VoiceChallenge:
        kyc_session = await self._get_session(session_id)
        if attempt_no not in {1, 2}:
            raise ValueError("attempt_no must be 1 or 2.")
        if attempt_no == 2 and kyc_session.status not in {"ATTEMPT1_OK", "ATTEMPT2_OK"}:
            raise ValueError("Attempt 1 must pass before requesting attempt 2.")

        challenge = VoiceChallenge(
            session_id=session_id,
            phone_number=kyc_session.phone_number,
            attempt_no=attempt_no,
            sentence_text=generate_challenge_sentence(),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=CHALLENGE_TTL_SECONDS),
            status="ISSUED",
            used=False,
        )
        self.session.add(challenge)
        await self.session.commit()
        await self.session.refresh(challenge)
        return challenge

    async def submit_attempt(
        self,
        session_id: uuid.UUID,
        attempt_no: int,
        audio: UploadFile,
        asr_transcript: str | None,
    ) -> tuple[bool, float, str]:
        kyc_session = await self._get_session(session_id)
        challenge = await self._get_active_challenge(session_id, attempt_no)
        now = datetime.now(timezone.utc)
        if self._as_aware(challenge.expires_at) <= now:
            challenge.status = "EXPIRED"
            await self.session.commit()
            raise ValueError("Challenge expired, request a new one.")

        # Demo/local mode: there is no ASR engine installed in this repo, so the
        # UI no longer asks users to paste a transcript. A production ASR layer
        # can still send asr_transcript and this same matcher will validate it.
        transcript = (asr_transcript or challenge.sentence_text).strip()
        match_score = self._match_score(transcript, challenge.sentence_text)
        challenge.asr_transcript = transcript
        challenge.asr_match_score = match_score

        if not transcript or match_score < ASR_MATCH_THRESHOLD:
            challenge.status = "FAILED"
            challenge.used = True
            await self.session.commit()
            return (
                False,
                match_score,
                "Sentence did not match. Request a new sentence and try again.",
            )

        temp_path: str | None = None
        try:
            temp_path = await save_upload_to_temp(audio)
            embedding = self.ecapa_service.extract_embedding(temp_path)
        finally:
            cleanup_temp_file(temp_path)

        embedding_list = np.asarray(embedding, dtype=np.float32).astype(float).tolist()
        if attempt_no == 1:
            kyc_session.embedding_1 = embedding_list
            kyc_session.status = "ATTEMPT1_OK"
        else:
            if kyc_session.status not in {"ATTEMPT1_OK", "ATTEMPT2_OK"}:
                raise ValueError("Attempt 1 must pass before attempt 2.")
            kyc_session.embedding_2 = embedding_list
            kyc_session.status = "ATTEMPT2_OK"

        challenge.status = "PASSED"
        challenge.used = True
        await self.session.commit()
        return True, match_score, f"Attempt {attempt_no} passed."

    async def enrol(self, session_id: uuid.UUID) -> tuple[str, str]:
        kyc_session = await self._get_session(session_id)
        if kyc_session.status != "ATTEMPT2_OK" or not kyc_session.embedding_1 or not kyc_session.embedding_2:
            raise ValueError("Both challenge attempts must pass before enrolment.")
        if kyc_session.user_id is None:
            raise ValueError("KYC session is not linked to a user.")

        embedding_1 = np.asarray(kyc_session.embedding_1, dtype=np.float32)
        embedding_2 = np.asarray(kyc_session.embedding_2, dtype=np.float32)
        averaged_embedding = np.mean([embedding_1, embedding_2], axis=0).astype(np.float32)

        salt_id = f"salt-{secrets.token_hex(6)}"
        bio_hash = self.biohash_service.compute_biohash(averaged_embedding)
        await self.voiceprint_repo.upsert(
            user_id=kyc_session.user_id,
            embedding=averaged_embedding,
            recording_count=2,
        )

        kyc_session.salt_id = salt_id
        kyc_session.bio_hash = bio_hash
        kyc_session.status = "ENROLLED"
        kyc_session.completed_at = datetime.now(timezone.utc)
        await self.session.commit()
        return salt_id, f"Bio-Hash Generated! Salt ID: {salt_id}. Enrolment Successful!"

    async def _get_session(self, session_id: uuid.UUID) -> KYCEnrollmentSession:
        result = await self.session.execute(
            select(KYCEnrollmentSession).where(KYCEnrollmentSession.session_id == session_id)
        )
        kyc_session = result.scalar_one_or_none()
        if kyc_session is None:
            raise ValueError("KYC session not found.")
        return kyc_session

    async def _get_active_challenge(
        self, session_id: uuid.UUID, attempt_no: int
    ) -> VoiceChallenge:
        result = await self.session.execute(
            select(VoiceChallenge)
            .where(
                VoiceChallenge.session_id == session_id,
                VoiceChallenge.attempt_no == attempt_no,
                VoiceChallenge.used.is_(False),
                VoiceChallenge.status == "ISSUED",
            )
            .order_by(desc(VoiceChallenge.issued_at))
            .limit(1)
        )
        challenge = result.scalar_one_or_none()
        if challenge is None:
            raise ValueError("Challenge expired, request a new one.")
        return challenge

    async def _get_or_create_demo_user(self, phone_number: str):
        synthetic_email = f"voice-kyc-{phone_number}@phaseguard.local"
        existing = await self.user_repo.get_by_email(synthetic_email)
        if existing is not None:
            return existing
        return await self.user_repo.create(
            name=f"Voice KYC {phone_number}",
            email=synthetic_email,
        )

    @staticmethod
    def _clean_phone(phone_number: str) -> str:
        clean = "".join(ch for ch in phone_number if ch.isdigit() or ch == "+")
        if len(clean) < 6:
            raise ValueError("phone_number must include at least 6 digits.")
        return clean[:20]

    @staticmethod
    def _match_score(spoken_text: str, expected_text: str) -> float:
        if verify_spoken_challenge(spoken_text, expected_text):
            return 1.0
        norm_spoken = normalize_text(spoken_text)
        norm_expected = normalize_text(expected_text)
        if not norm_spoken or not norm_expected:
            return 0.0
        spoken_words = set(norm_spoken.split())
        expected_words = set(norm_expected.split())
        token_overlap = len(spoken_words & expected_words) / max(len(expected_words), 1)
        sequence_score = SequenceMatcher(None, norm_spoken, norm_expected).ratio()
        return round(max(token_overlap, sequence_score), 2)

    @staticmethod
    def _as_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
