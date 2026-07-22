from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings


REPORT_CSV = Path("evaluation/reports/faiss_scaling_benchmark.csv")
REPORT_JSON = Path("evaluation/reports/faiss_scaling_benchmark.json")
DESIGN_MD = Path("evaluation/reports/faiss_multi_instance_design.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local FAISS search latency and recall.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--index-types", nargs="+", default=["flat", "hnsw", "ivfpq"])
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    blockers = []
    for size in args.sizes:
        vectors, queries = make_vectors(size, args.queries, args.seed)
        exact = build_flat(vectors)
        _, exact_ids = exact.search(queries, args.k)
        for index_type in args.index_types:
            try:
                index = build_index(index_type, vectors)
                metrics = time_search(index, queries, exact_ids, args.k)
                rows.append({"index_type": index_type, "vector_count": size, **metrics})
                print(f"{index_type} {size}: p95={metrics['p95_ms']:.3f}ms recall@{args.k}={metrics['recall_at_k']:.4f}")
            except Exception as exc:  # noqa: BLE001
                blockers.append({"index_type": index_type, "vector_count": size, "reason": repr(exc)})

    write_csv(rows)
    REPORT_JSON.write_text(
        json.dumps(
            {
                "status": "complete" if not blockers else "partial",
                "rows_csv": str(REPORT_CSV),
                "blockers": blockers,
                "note": "Local synthetic-vector benchmark on this workstation; not a production guarantee.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_design()
    return 0 if rows else 1


def make_vectors(size: int, queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + size)
    dim = get_settings().EMBEDDING_DIM
    vectors = rng.normal(size=(size, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    query_ids = rng.choice(size, size=min(queries, size), replace=False)
    q = vectors[query_ids].copy()
    q += rng.normal(scale=0.01, size=q.shape).astype(np.float32)
    faiss.normalize_L2(q)
    return vectors, q


def build_flat(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_index(index_type: str, vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    settings = get_settings()
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = 80
    elif index_type == "ivfpq":
        nlist = max(16, min(4096, int(np.sqrt(len(vectors)))))
        quantizer = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, settings.FAISS_M, settings.FAISS_NBITS, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = min(settings.FAISS_NPROBE, nlist)
        index.train(vectors)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    index.add(vectors)
    return index


def time_search(index: faiss.Index, queries: np.ndarray, exact_ids: np.ndarray, k: int) -> dict[str, float]:
    latencies = []
    recalls = []
    for query, exact in zip(queries, exact_ids):
        q = query.reshape(1, -1)
        started = time.perf_counter()
        _, ids = index.search(q, k)
        latencies.append((time.perf_counter() - started) * 1000.0)
        recalls.append(len(set(ids[0]) & set(exact)) / k)
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "queries": len(queries),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "recall_at_k": float(np.mean(recalls)),
    }


def write_csv(rows: list[dict]) -> None:
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["index_type", "vector_count", "queries", "p50_ms", "p95_ms", "p99_ms", "recall_at_k"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_design() -> None:
    DESIGN_MD.write_text(
        """# FAISS Multi-Instance Index Strategy

Current app instances keep FAISS in process memory, so multiple API nodes do not share live index state.

Recommended production design:

1. Treat PostgreSQL encrypted voiceprints as the source of truth.
2. Build FAISS snapshots in a scheduled single-writer job from decrypted in-memory vectors.
3. Publish each snapshot plus mapping JSON to shared object storage with a version manifest.
4. Each API instance polls the manifest, downloads a complete new snapshot, validates checksum, then atomically swaps the in-memory index.
5. Enrollment writes update PostgreSQL immediately and optionally pushes a small event; API nodes either apply short-lived incremental updates or wait for the next snapshot depending on consistency requirements.
6. Keep old snapshots until every node reports the new version.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
