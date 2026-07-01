"""
app/encryption/aes_gcm.py

AES-256-GCM authenticated encryption for voice biometric templates.

Why AES-256-GCM:
  - 256-bit key: quantum-resistant for the foreseeable future
  - GCM mode: provides both confidentiality AND integrity (authentication tag)
  - Authenticated: detects any tampering with the ciphertext
  - Random 96-bit nonce per operation: prevents nonce reuse attacks

Usage in PhaseGuard:
  - Encrypt cancelable BioHash-transformed embeddings before DB storage
  - Encrypt emergency contact phone numbers
  - Encrypt consent tokens at rest

Security properties:
  - A compromised ciphertext cannot be decrypted without the key
  - Any bit-flip in the ciphertext is detected by the authentication tag
  - Different nonces for each encrypt call prevent ciphertext comparison attacks
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_BYTES = 12   # 96-bit nonce (GCM standard)
KEY_BYTES = 32     # 256-bit key
TAG_BYTES = 16     # 128-bit authentication tag (GCM default)


@dataclass
class EncryptedData:
    """Container for AES-256-GCM encrypted data."""
    ciphertext: bytes    # Encrypted payload (includes authentication tag)
    nonce: bytes         # 96-bit random nonce used for this operation


class AESGCM256:
    """
    AES-256-GCM encryption/decryption wrapper.

    Args:
        key: 32-byte (256-bit) encryption key. MUST be kept secret.
             In production: retrieved from HSM via KeyManager.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) != KEY_BYTES:
            raise ValueError(f"Key must be {KEY_BYTES} bytes, got {len(key)}")
        self._aesgcm = AESGCM(key)

    def encrypt(
        self,
        plaintext: bytes,
        associated_data: bytes | None = None,
    ) -> EncryptedData:
        """
        Encrypt plaintext with AES-256-GCM.

        Args:
            plaintext: Data to encrypt (e.g., serialized embedding bytes)
            associated_data: Optional authenticated-but-not-encrypted data
                             (e.g., user_id bytes for binding the ciphertext
                              to a specific user — prevents ciphertext swap attacks)

        Returns:
            EncryptedData with ciphertext (+ 16-byte GCM tag) and random nonce
        """
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return EncryptedData(ciphertext=ciphertext, nonce=nonce)

    def decrypt(
        self,
        encrypted: EncryptedData,
        associated_data: bytes | None = None,
    ) -> bytes:
        """
        Decrypt AES-256-GCM ciphertext.

        Args:
            encrypted: EncryptedData containing ciphertext and nonce
            associated_data: Must match what was passed during encryption

        Returns:
            Decrypted plaintext bytes

        Raises:
            cryptography.exceptions.InvalidTag: If authentication tag verification
                fails (ciphertext tampered or wrong key)
        """
        return self._aesgcm.decrypt(
            encrypted.nonce, encrypted.ciphertext, associated_data
        )


def generate_key() -> bytes:
    """Generate a new cryptographically secure 256-bit AES key."""
    return os.urandom(KEY_BYTES)


def key_from_string(key_str: str) -> bytes:
    """
    Derive a 32-byte key from a string (e.g., from environment variable).

    Uses first 32 bytes of the UTF-8 encoded string, padded with zeros if shorter.
    For production: use a proper key derivation function (HKDF) instead.
    """
    raw = key_str.encode("utf-8")
    if len(raw) < KEY_BYTES:
        raw = raw + b"\x00" * (KEY_BYTES - len(raw))
    return raw[:KEY_BYTES]
