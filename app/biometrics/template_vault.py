"""
app/biometrics/template_vault.py

Template vault and lifecycle manager.

Orchestrates template generation, verification, rotation, and revocation:
  1. Retrieve user-specific key from KeyManager
  2. Compute BioHash transformation
  3. Encrypt using AES-256-GCM
  4. Write/read template blobs
  5. Compare incoming challenge template against vault templates

Security rules:
  - Raw embeddings are NEVER written to the database. They exist only
    in CPU register memory during transaction lifecycle.
  - A template hash is maintained for database row integrity verification.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
import numpy as np

from app.biometrics.biohash import get_biohash
from app.core.config import get_settings
from app.encryption.aes_gcm import AESGCM256, EncryptedData
from app.encryption.key_manager import get_key_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class TemplateVault:
    """Manages the creation, verification, and rotation of cancelable biometrics."""

    def __init__(self) -> None:
        self.key_manager = get_key_manager()
        self.biohash = get_biohash()

    def generate_template(
        self,
        user_id: uuid.UUID,
        raw_embedding: np.ndarray,
    ) -> tuple[bytes, bytes, str, str]:
        """
        Generate an encrypted cancelable template from a raw embedding.

        Args:
            user_id: The target user
            raw_embedding: Raw float32 array

        Returns:
            Tuple of:
              - encrypted_template_bytes
              - nonce_bytes
              - sha256_hash_hex
              - key_id used
        """
        # 1. Retrieve user-specific derived key
        key_id = settings.HSM_MASTER_KEY_ID
        user_key = self.key_manager.get_key_for_user(str(user_id), key_id)

        # 2. Apply BioHash transformation
        hashed_embedding = self.biohash.transform(raw_embedding, user_key)

        # 3. Serialize to bytes
        flat_bytes = hashed_embedding.tobytes()

        # 4. Generate integrity hash of the plaintext template
        integrity_hash = hashlib.sha256(flat_bytes).hexdigest()

        # 5. Encrypt template using AES-GCM bound to user_id (Associated Data)
        encryptor = AESGCM256(user_key)
        encrypted = encryptor.encrypt(flat_bytes, associated_data=str(user_id).encode())

        logger.debug(
            "Template generated for user %s, key_id=%s, integrity_hash=%s",
            user_id, key_id, integrity_hash[:8]
        )

        return encrypted.ciphertext, encrypted.nonce, integrity_hash, key_id

    def decrypt_template(
        self,
        user_id: uuid.UUID,
        encrypted_embedding: bytes,
        nonce: bytes,
        key_id: str,
    ) -> np.ndarray:
        """
        Decrypt and reconstruct the BioHash template.

        Args:
            user_id: Target user
            encrypted_embedding: Ciphertext from database
            nonce: IV used during encryption
            key_id: HSM key ID

        Returns:
            BioHashed embedding as 1D float32 numpy array
        """
        # 1. Retrieve key
        user_key = self.key_manager.get_key_for_user(str(user_id), key_id)

        # 2. Decrypt
        encryptor = AESGCM256(user_key)
        encrypted_data = EncryptedData(ciphertext=encrypted_embedding, nonce=nonce)
        decrypted_bytes = encryptor.decrypt(encrypted_data, associated_data=str(user_id).encode())

        # 3. Deserialise
        template = np.frombuffer(decrypted_bytes, dtype=np.float32).copy()

        # Verify dimension
        expected_dim = settings.BIOHASH_PROJECTION_DIM
        if len(template) != expected_dim:
            raise ValueError(f"Template dimension mismatch: expected {expected_dim}, got {len(template)}")

        return template

    def generate_challenge_template(
        self,
        user_id: uuid.UUID,
        raw_embedding: np.ndarray,
        key_id: str,
    ) -> np.ndarray:
        """
        Transform a live verification embedding into a BioHash template.

        Unlike generate_template(), this returns the plaintext BioHash vector
        for direct comparison in memory (not encrypted, not stored in DB).
        """
        user_key = self.key_manager.get_key_for_user(str(user_id), key_id)
        return self.biohash.transform(raw_embedding, user_key)


# Module-level singleton
_vault_instance: TemplateVault | None = None


def get_template_vault() -> TemplateVault:
    """Return singleton TemplateVault."""
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = TemplateVault()
    return _vault_instance
