from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


REPORT_PATH = Path("evaluation/reports/hf_kathbath_access_check.json")


def main() -> int:
    load_dotenv(".env", override=False)
    token_loaded = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    payload = {
        "status": "unknown",
        "hf_token_loaded": token_loaded,
        "dataset": "ai4bharat/Kathbath",
        "config": "hindi",
        "split": "valid",
        "checked_at_unix": time.time(),
    }
    try:
        import datasets.features.audio as fa

        fa.Audio.decode_example = lambda self, value, token_per_repo_id=None: value

        from datasets import load_dataset

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        dataset = load_dataset(
            "ai4bharat/Kathbath",
            "hindi",
            split="valid",
            streaming=True,
            token=token,
        )
        row = next(iter(dataset))
        payload.update(
            {
                "status": "ok",
                "sample_fields": sorted(row.keys()),
                "sample_speaker_id_present": row.get("speaker_id") is not None,
                "sample_gender_present": row.get("gender") is not None,
                "sample_language": row.get("lang"),
            }
        )
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "blocked",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        )
        exit_code = 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
