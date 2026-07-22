"""
train_faiss_index.py

Offline training script for scalable FAISS IndexIVFPQ + IndexHNSWFlat coarse quantizer.
Trains quantizer using unlabelled voiceprints and saves trained index file to disk.
"""

import os
import faiss
import numpy as np
from app.core.config import get_settings

settings = get_settings()


def train_index():
    dimension = settings.EMBEDDING_DIM
    m = settings.FAISS_M
    nbits = settings.FAISS_NBITS
    nlist = 256  # Voronoi cells

    print(f"Initializing FAISS IVF-PQ index (dimension={dimension}, M={m}, nbits={nbits})...")

    # Coarse quantizer using HNSW
    quantizer = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    # Generate synthetic training vectors for offline index training
    rng = np.random.RandomState(42)
    num_training_vectors = 10000
    training_data = rng.randn(num_training_vectors, dimension).astype(np.float32)
    faiss.normalize_L2(training_data)

    print(f"Training IVF-PQ quantizer on {num_training_vectors} vectors...")
    index.train(training_data)
    print("Training complete. Index is trained:", index.is_trained)

    index_dir = os.path.dirname(settings.FAISS_INDEX_PATH)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, settings.FAISS_INDEX_PATH)
    print(f"Saved trained FAISS index to {settings.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    train_index()
