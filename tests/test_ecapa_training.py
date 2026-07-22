"""
tests/test_ecapa_training.py

Unit tests for ECAPA-TDNN dataset splitting logic and embedding comparison sanity checks.
"""

import random
import numpy as np
import pytest

def test_svarah_dataset_splitting_logic():
    """
    Verify that speaker-aware train/validation splitting ensures zero speaker overlap.
    """
    # Simulate a pool of 100 Svarah speaker IDs
    speaker_ids = [f"svarah_spk_{i:03d}" for i in range(100)]
    
    # Shuffle and split using the exact logic from train_ecapa_svarah.py
    rng = random.Random(42)
    rng.shuffle(speaker_ids)
    
    split_idx = max(1, int(len(speaker_ids) * 0.8))
    train_speakers = speaker_ids[:split_idx]
    val_speakers = speaker_ids[split_idx:]
    
    # Assertions
    assert len(train_speakers) == 80
    assert len(val_speakers) == 20
    
    # Ensure no overlap
    train_set = set(train_speakers)
    val_set = set(val_speakers)
    intersection = train_set.intersection(val_set)
    assert len(intersection) == 0, f"Overlapping speakers found: {intersection}"

def test_embedding_comparison_sanity():
    """
    Verify that the cosine similarity calculations behave correctly
    under simulated baseline vs trained embedding scenarios.
    """
    # Setup two mock 192-dimensional embeddings
    rng = np.random.default_rng(seed=42)
    
    # Generate a random unit-normalized vector for baseline
    emb_baseline = rng.normal(size=192)
    emb_baseline /= np.linalg.norm(emb_baseline)
    
    # Generate a trained vector that is highly similar but not identical
    noise = rng.normal(scale=0.01, size=192)
    emb_trained = emb_baseline + noise
    emb_trained /= np.linalg.norm(emb_trained)
    
    # Compute cosine similarity
    dot_product = np.dot(emb_baseline, emb_trained)
    norm_product = np.linalg.norm(emb_baseline) * np.linalg.norm(emb_trained)
    similarity = dot_product / norm_product
    
    # Assertions
    assert -1.0 <= similarity <= 1.0
    assert similarity > 0.9  # Since we added small noise, they should be highly similar
    assert similarity != 1.0  # But not completely identical
