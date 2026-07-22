import base64
import uuid

import numpy as np

from app.models.voiceprint import Voiceprint
from app.services.biohash_service import BioHashService
from app.services.voiceprint_crypto import VOICEPRINT_KEY_ENV, VoiceprintCrypto


def test_voiceprint_crypto_round_trips_embedding_and_biohash(monkeypatch):
    key = bytes(range(32))
    monkeypatch.setenv(VOICEPRINT_KEY_ENV, base64.b64encode(key).decode("ascii"))
    crypto = VoiceprintCrypto.from_env()

    embedding = np.linspace(-1.0, 1.0, 192, dtype=np.float32)
    biohash = BioHashService().compute_biohash(embedding)

    encrypted_embedding = crypto.encrypt_embedding(embedding)
    encrypted_biohash = crypto.encrypt_biohash(biohash)

    assert encrypted_embedding.ciphertext
    assert encrypted_embedding.nonce
    assert encrypted_biohash is not None
    assert crypto.decrypt_embedding(encrypted_embedding.ciphertext, encrypted_embedding.nonce) == embedding.astype(float).tolist()
    assert crypto.decrypt_biohash(encrypted_biohash.ciphertext, encrypted_biohash.nonce) == biohash


def test_voiceprint_model_can_hold_plaintext_without_raw_db_columns():
    embedding = np.linspace(-1.0, 1.0, 192, dtype=np.float32).astype(float).tolist()
    voiceprint = Voiceprint(
        user_id=uuid.uuid4(),
        encrypted_embedding=b"ciphertext",
        embedding_nonce=b"123456789012",
        embedding_key_id="test",
        recording_count=3,
        model_version="unit-test",
    )

    voiceprint._raw_embedding_column = None
    voiceprint._raw_biohash_column = None
    voiceprint.attach_plaintext(embedding, "0" * 256)

    assert voiceprint.embedding == embedding
    assert voiceprint.biohash == "0" * 256
    assert voiceprint._raw_embedding_column is None
    assert voiceprint._raw_biohash_column is None
