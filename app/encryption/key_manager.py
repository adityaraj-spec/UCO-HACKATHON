"""
app/encryption/key_manager.py

Abstract key management interface and concrete implementations.

Hierarchy:
  KeyManager (ABC)
    ├── MockHSMKeyManager    (in-memory, for development)
    └── AWSCloudHSMKeyManager (stub for production)

The KeyManager abstraction decouples the application from any specific
HSM vendor. Switching from mock to AWS CloudHSM requires only changing
HSM_PROVIDER=aws_cloudhsm in the environment — no code changes.

In production banking:
  - Keys NEVER leave the HSM boundary
  - All cryptographic operations happen inside the HSM
  - This implementation approximates this by keeping keys in memory
    (development) or calling the HSM API (production stub)
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

KEY_SIZE_BYTES = 32  # AES-256


@dataclass
class KeyInfo:
    """Metadata about a managed key."""
    key_id: str
    version: int
    created_at: datetime
    is_active: bool


class KeyManager(ABC):
    """
    Abstract key management interface.

    All implementations must be thread-safe.
    """

    @abstractmethod
    def get_key(self, key_id: str) -> bytes:
        """
        Retrieve a key by ID.

        Args:
            key_id: Unique key identifier

        Returns:
            Raw key bytes (32 bytes for AES-256)

        Raises:
            KeyError: If key_id does not exist
        """

    @abstractmethod
    def create_key(self, key_id: str) -> str:
        """
        Create a new key with the given ID.

        Args:
            key_id: Unique identifier for the new key

        Returns:
            The key_id (same as input, for chaining)
        """

    @abstractmethod
    def rotate_key(self, key_id: str) -> str:
        """
        Rotate a key, creating a new version.

        The old key version should be retained for decryption of existing data
        until all data has been re-encrypted with the new key.

        Args:
            key_id: Key to rotate

        Returns:
            New key_id (typically key_id + version suffix)
        """

    @abstractmethod
    def list_key_ids(self) -> list[str]:
        """List all key IDs managed by this instance."""

    def get_key_for_user(self, user_id: str, master_key_id: str) -> bytes:
        """
        Derive a per-user key from the master key.

        Uses HKDF-SHA256 with user_id as info parameter.
        This ensures each user's data is encrypted with a distinct key,
        limiting the blast radius of any single key compromise.
        """
        master_key = self.get_key(master_key_id)
        # HKDF: Derive per-user key
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF

            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=f"phaseguard-user-{user_id}".encode(),
            )
            return hkdf.derive(master_key)
        except ImportError:
            # Fallback: HMAC-SHA256 key derivation
            import hmac
            return hmac.new(
                master_key,
                f"phaseguard-user-{user_id}".encode(),
                hashlib.sha256,
            ).digest()


class MockHSMKeyManager(KeyManager):
    """
    In-memory mock HSM key manager for development and testing.

    WARNING: Keys are stored in process memory. This is NOT production-safe.
    For production: replace with AWSCloudHSMKeyManager or PKCS#11 provider.

    The mock accepts a pre-configured master key from environment variables
    so tests and Docker environments can use deterministic keys.
    """

    def __init__(self, initial_master_key: bytes | None = None) -> None:
        self._keys: dict[str, bytes] = {}
        self._versions: dict[str, int] = {}

        # Load or generate master key
        if initial_master_key is not None:
            if len(initial_master_key) != KEY_SIZE_BYTES:
                raise ValueError("Master key must be 32 bytes")
            master_key = initial_master_key
        else:
            master_key = secrets.token_bytes(KEY_SIZE_BYTES)

        # Pre-load the PhaseGuard master key
        master_key_id = "phaseguard-master-key-v1"
        self._keys[master_key_id] = master_key
        self._versions[master_key_id] = 1
        logger.info("MockHSM initialized with master key ID: %s", master_key_id)

    def get_key(self, key_id: str) -> bytes:
        if key_id not in self._keys:
            raise KeyError(f"Key not found: {key_id}")
        return self._keys[key_id]

    def create_key(self, key_id: str) -> str:
        if key_id in self._keys:
            logger.warning("Key already exists, returning existing: %s", key_id)
            return key_id
        self._keys[key_id] = secrets.token_bytes(KEY_SIZE_BYTES)
        self._versions[key_id] = 1
        logger.info("MockHSM: created key %s", key_id)
        return key_id

    def rotate_key(self, key_id: str) -> str:
        if key_id not in self._keys:
            raise KeyError(f"Cannot rotate non-existent key: {key_id}")
        old_version = self._versions.get(key_id, 1)
        new_version = old_version + 1
        new_key_id = f"{key_id.rsplit('-', 1)[0]}-v{new_version}"
        self._keys[new_key_id] = secrets.token_bytes(KEY_SIZE_BYTES)
        self._versions[new_key_id] = new_version
        logger.info("MockHSM: rotated %s → %s", key_id, new_key_id)
        return new_key_id

    def list_key_ids(self) -> list[str]:
        return list(self._keys.keys())


class AWSCloudHSMKeyManager(KeyManager):
    """
    AWS CloudHSM key manager stub.

    Implements the KeyManager interface for AWS CloudHSM via the PKCS#11
    interface. In a real deployment:
      1. Install aws-cloudhsm-client and cloudhsm-pkcs11 packages
      2. Configure HSM cluster endpoint and credentials
      3. Replace stub methods with actual CloudHSM API calls

    This stub raises NotImplementedError to clearly signal that real HSM
    integration requires additional configuration.
    """

    def __init__(self, cluster_id: str, region: str) -> None:
        self.cluster_id = cluster_id
        self.region = region
        logger.warning(
            "AWSCloudHSMKeyManager stub initialized. "
            "Real HSM integration requires additional configuration."
        )

    def get_key(self, key_id: str) -> bytes:
        raise NotImplementedError(
            "AWS CloudHSM integration not configured. "
            "Set HSM_PROVIDER=mock for development or configure PKCS#11."
        )

    def create_key(self, key_id: str) -> str:
        raise NotImplementedError("AWS CloudHSM integration not configured.")

    def rotate_key(self, key_id: str) -> str:
        raise NotImplementedError("AWS CloudHSM integration not configured.")

    def list_key_ids(self) -> list[str]:
        return []


# ------------------------------------------------------------------ #
# Factory                                                              #
# ------------------------------------------------------------------ #

_key_manager: KeyManager | None = None


def get_key_manager() -> KeyManager:
    """
    Return the configured KeyManager singleton.

    Selection is based on HSM_PROVIDER environment variable:
      mock         → MockHSMKeyManager (development, default)
      aws_cloudhsm → AWSCloudHSMKeyManager (production)
    """
    global _key_manager
    if _key_manager is not None:
        return _key_manager

    from app.core.config import get_settings
    settings = get_settings()

    if settings.HSM_PROVIDER == "mock":
        from app.encryption.aes_gcm import key_from_string
        master_key = key_from_string(settings.MOCK_HSM_KEY)
        _key_manager = MockHSMKeyManager(initial_master_key=master_key)
    elif settings.HSM_PROVIDER == "aws_cloudhsm":
        _key_manager = AWSCloudHSMKeyManager(
            cluster_id=getattr(settings, "AWS_HSM_CLUSTER_ID", "UNSET"),
            region=getattr(settings, "AWS_REGION", "ap-south-1"),
        )
    else:
        raise ValueError(f"Unknown HSM_PROVIDER: {settings.HSM_PROVIDER}")

    return _key_manager
