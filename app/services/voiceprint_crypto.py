"""
app/services/voiceprint_crypto.py

AES-256-GCM helpers for encrypting durable voiceprint material before it is
written to PostgreSQL. The raw embedding remains available only in process
for verification and FAISS cache updates.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


VOICEPRINT_KEY_ENV = "VOICEPRINT_ENCRYPTION_KEY_B64"
VOICEPRINT_KEY_ID_ENV = "VOICEPRINT_ENCRYPTION_KEY_ID"


@dataclass(frozen=True)
class EncryptedValue:
    ciphertext: bytes
    nonce: bytes
    key_id: str


class VoiceprintCrypto:
    """Encrypts/decrypts embeddings and BioHash strings with AES-256-GCM."""

    def __init__(self, key: bytes, key_id: str = "default") -> None:
        if len(key) != 32:
            raise ValueError("Voiceprint encryption key must be exactly 32 bytes for AES-256-GCM.")
        self.key = key
        self.key_id = key_id
        self._aesgcm = AESGCM(key)

    @classmethod
    def from_env(cls) -> "VoiceprintCrypto":
        encoded = os.environ.get(VOICEPRINT_KEY_ENV)
        if not encoded:
            raise RuntimeError(f"{VOICEPRINT_KEY_ENV} is required to encrypt voiceprints at rest.")
        key = base64.b64decode(encoded)
        key_id = os.environ.get(VOICEPRINT_KEY_ID_ENV, "default")
        return cls(key=key, key_id=key_id)

    def encrypt_embedding(self, embedding: list[float] | np.ndarray) -> EncryptedValue:
        arr = np.asarray(embedding, dtype=np.float32).flatten()
        payload = json.dumps(arr.astype(float).tolist(), separators=(",", ":")).encode("utf-8")
        return self._encrypt(payload, b"voiceprint.embedding")

    def decrypt_embedding(self, ciphertext: bytes, nonce: bytes) -> list[float]:
        payload = self._aesgcm.decrypt(nonce, ciphertext, b"voiceprint.embedding")
        return json.loads(payload.decode("utf-8"))

    def encrypt_biohash(self, biohash: str | None) -> EncryptedValue | None:
        if biohash is None:
            return None
        return self._encrypt(biohash.encode("utf-8"), b"voiceprint.biohash")

    def decrypt_biohash(self, ciphertext: bytes | None, nonce: bytes | None) -> str | None:
        if ciphertext is None or nonce is None:
            return None
        payload = self._aesgcm.decrypt(nonce, ciphertext, b"voiceprint.biohash")
        return payload.decode("utf-8")

    def _encrypt(self, payload: bytes, aad: bytes) -> EncryptedValue:
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, payload, aad)
        return EncryptedValue(ciphertext=ciphertext, nonce=nonce, key_id=self.key_id)
