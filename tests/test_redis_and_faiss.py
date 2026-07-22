"""
tests/test_redis_and_faiss.py

Unit tests for RedisService (rate limiting + replay protection) and FAISSService (top-k search).
Forces in-memory fallback mode for Redis during tests.
"""

import time
import uuid
import pytest
from app.services.redis_service import RedisService
from app.services.faiss_service import FAISSService

def test_redis_service_in_memory_fallback(monkeypatch):
    """
    Test RedisService features using the thread-safe in-memory fallback.
    """
    # Prevent real connection attempts during unit test
    monkeypatch.setattr(RedisService, "_connect", lambda self: None)
    
    service = RedisService()
    service.is_connected = False
    service.redis_client = None
    
    # 1. Test basic Get/Set
    assert service.set("test_key", "hello", ex=60) is True
    assert service.get("test_key") == "hello"
    
    # 2. Test Key Expiry
    service.set("expiring_key", "value", ex=1)
    time.sleep(1.1)
    assert service.get("expiring_key") is None
    
    # 3. Test JWT Replay Protection
    jti = "unique-jti-token-123"
    # First time marking it used should succeed
    assert service.mark_token_used(jti, ttl_seconds=10) is True
    # Replaying it should fail
    assert service.mark_token_used(jti, ttl_seconds=10) is False
    
    # 4. Test Rate Limiting
    key = "user_rate_limit"
    # Limit: 3 requests per 2 seconds
    assert service.is_rate_limited(key, limit=3, window_seconds=2) is False
    assert service.is_rate_limited(key, limit=3, window_seconds=2) is False
    assert service.is_rate_limited(key, limit=3, window_seconds=2) is False
    # 4th request exceeds limit
    assert service.is_rate_limited(key, limit=3, window_seconds=2) is True
    
    # Wait for window to clear
    time.sleep(2.1)
    assert service.is_rate_limited(key, limit=3, window_seconds=2) is False


def test_faiss_service_operations(tmp_path):
    """
    Test FAISSService indexing, searching, updating and persistence.
    """
    index_dir = str(tmp_path)
    service = FAISSService(index_dir=index_dir)
    
    # Generate mock orthogonal embeddings (192-dim) so they have distinct directions
    emb_user_a = [0.0] * 192
    emb_user_a[0] = 1.0
    
    emb_user_b = [0.0] * 192
    emb_user_b[1] = 1.0
    
    user_a_uuid = uuid.uuid4()
    user_b_uuid = uuid.uuid4()
    
    # Add to index
    service.add_or_update(user_a_uuid, emb_user_a)
    service.add_or_update(user_b_uuid, emb_user_b)
    
    assert service.index.ntotal == 2
    
    # Search for user_a using user_a's embedding (should return user_a as top match)
    matches = service.search(emb_user_a, k=1)
    assert len(matches) == 1
    matched_uuid, score = matches[0]
    assert matched_uuid == user_a_uuid
    assert score > 0.99  # Perfect inner product for unit vector
    
    # Test persistence and recovery
    service.save_to_disk()
    
    # Initialize a new service instance reading from same temp dir
    recovered_service = FAISSService(index_dir=index_dir)
    assert recovered_service.index.ntotal == 2
    
    # Search with recovered service
    recovered_matches = recovered_service.search(emb_user_b, k=1)
    assert len(recovered_matches) == 1
    assert recovered_matches[0][0] == user_b_uuid


def test_faiss_service_hnsw_index_type(tmp_path):
    service = FAISSService(index_dir=str(tmp_path), index_type="hnsw")

    emb_user_a = [0.0] * 192
    emb_user_a[0] = 1.0
    emb_user_b = [0.0] * 192
    emb_user_b[1] = 1.0
    user_a_uuid = uuid.uuid4()
    user_b_uuid = uuid.uuid4()

    service.add_or_update(user_a_uuid, emb_user_a)
    service.add_or_update(user_b_uuid, emb_user_b)

    matches = service.search(emb_user_a, k=1)
    assert len(matches) == 1
    assert matches[0][0] == user_a_uuid
    assert matches[0][1] > 0.9
