# FAISS Multi-Instance Index Strategy

Current app instances keep FAISS in process memory, so multiple API nodes do not share live index state.

Recommended production design:

1. Treat PostgreSQL encrypted voiceprints as the source of truth.
2. Build FAISS snapshots in a scheduled single-writer job from decrypted in-memory vectors.
3. Publish each snapshot plus mapping JSON to shared object storage with a version manifest.
4. Each API instance polls the manifest, downloads a complete new snapshot, validates checksum, then atomically swaps the in-memory index.
5. Enrollment writes update PostgreSQL immediately and optionally pushes a small event; API nodes either apply short-lived incremental updates or wait for the next snapshot depending on consistency requirements.
6. Keep old snapshots until every node reports the new version.
