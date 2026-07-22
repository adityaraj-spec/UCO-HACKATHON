from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import httpx


REPORT_JSON = Path("evaluation/reports/verify_latency_benchmark.json")
REPORT_CSV = Path("evaluation/reports/verify_latency_benchmark.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a live /api/v1/verify endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--audio-file", type=Path, required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 10, 50, 100])
    parser.add_argument("--requests-per-level", type=int, default=100)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if not args.audio_file.exists():
        write_blocked(f"Audio file does not exist: {args.audio_file}")
        return 1

    rows = []
    try:
        async with httpx.AsyncClient(timeout=args.timeout_sec) as client:
            for concurrency in args.concurrency:
                metrics = await run_level(client, args, concurrency)
                rows.append(metrics)
                print(f"c={concurrency}: p95={metrics['p95_ms']:.2f}ms errors={metrics['errors']}")
    except Exception as exc:  # noqa: BLE001
        write_blocked(str(exc))
        return 1

    if rows and all(row["errors"] == row["requests"] for row in rows):
        write_blocked("All benchmark requests failed; verify API is not reachable or has no valid enrolled user/audio context.")
        write_csv(rows)
        return 1

    write_csv(rows)
    REPORT_JSON.write_text(
        json.dumps(
            {
                "status": "complete",
                "rows_csv": str(REPORT_CSV),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


async def run_level(client: httpx.AsyncClient, args: argparse.Namespace, concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    stage_samples: dict[str, list[float]] = {}
    errors = 0

    async def one_request():
        nonlocal errors
        async with semaphore:
            started = time.perf_counter()
            try:
                with args.audio_file.open("rb") as handle:
                    response = await client.post(
                        f"{args.base_url}/api/v1/verify",
                        data={"user_id": args.user_id, "layer1_score": "0.0"},
                        files={"file": (args.audio_file.name, handle, "audio/wav")},
                )
                if response.status_code >= 400:
                    errors += 1
                else:
                    payload = response.json()
                    for stage, value in (payload.get("stage_timings_ms") or {}).items():
                        stage_samples.setdefault(stage, []).append(float(value))
            except Exception:
                errors += 1
            finally:
                latencies.append((time.perf_counter() - started) * 1000.0)

    await asyncio.gather(*(one_request() for _ in range(args.requests_per_level)))
    return summarise(concurrency, latencies, errors, stage_samples)


def summarise(
    concurrency: int,
    latencies: list[float],
    errors: int,
    stage_samples: dict[str, list[float]],
) -> dict:
    sorted_lat = sorted(latencies)
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "errors": errors,
        "p50_ms": percentile(sorted_lat, 50),
        "p95_ms": percentile(sorted_lat, 95),
        "p99_ms": percentile(sorted_lat, 99),
        "mean_ms": statistics.fmean(sorted_lat) if sorted_lat else 0.0,
        "stage_timings_ms": {
            stage: summarise_values(samples)
            for stage, samples in sorted(stage_samples.items())
        },
    }


def summarise_values(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        "p50": percentile(sorted_values, 50),
        "p95": percentile(sorted_values, 95),
        "p99": percentile(sorted_values, 99),
        "mean": statistics.fmean(sorted_values) if sorted_values else 0.0,
    }


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((p / 100.0) * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def write_csv(rows: list[dict]) -> None:
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "concurrency",
                "requests",
                "errors",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "mean_ms",
                "stage_timings_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["stage_timings_json"] = json.dumps(csv_row.pop("stage_timings_ms", {}), sort_keys=True)
            writer.writerow(csv_row)


def write_blocked(reason: str) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(
            {
                "status": "blocked",
                "reason": reason,
                "requires": "Running API, enrolled user_id, and representative audio file.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
