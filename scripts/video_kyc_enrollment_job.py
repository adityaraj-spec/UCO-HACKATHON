"""
Extract the voice-enrollment portion of a Video KYC recording and submit it
to the existing /api/v1/enroll endpoint.

Requires ffmpeg on PATH. The job cuts only the officer-marked 5-phrase
segment, splits it into phrase clips, converts each clip to 16 kHz mono WAV,
and sends those clips through the normal enrollment pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import requests


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_phrase_clips(
    video_path: Path,
    output_dir: Path,
    start_sec: float,
    end_sec: float,
    phrase_count: int,
) -> list[Path]:
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec.")
    if phrase_count <= 0:
        raise ValueError("phrase_count must be positive.")

    duration = end_sec - start_sec
    clip_duration = duration / phrase_count
    clips: list[Path] = []

    for index in range(phrase_count):
        clip_start = start_sec + (index * clip_duration)
        clip_path = output_dir / f"video_kyc_phrase_{index + 1}.wav"
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{clip_start:.3f}",
                "-t",
                f"{clip_duration:.3f}",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                str(clip_path),
            ]
        )
        clips.append(clip_path)

    return clips


def submit_enrollment(
    api_base_url: str,
    user_id: str,
    clips: list[Path],
    consent_confirmed: bool,
) -> requests.Response:
    files = [
        ("files", (clip.name, clip.read_bytes(), "audio/wav"))
        for clip in clips
    ]
    data = {
        "user_id": user_id,
        "channel": "VIDEO_KYC",
        "biometric_consent_confirmed": str(consent_confirmed).lower(),
        "identity_confirmed": "true",
    }
    return requests.post(
        f"{api_base_url.rstrip('/')}/api/v1/enroll",
        data=data,
        files=files,
        timeout=180,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Video KYC voice enrollment segment.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--start-sec", required=True, type=float)
    parser.add_argument("--end-sec", required=True, type=float)
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--phrase-count", default=5, type=int)
    parser.add_argument("--consent-confirmed", action="store_true")
    args = parser.parse_args()

    if not args.consent_confirmed:
        raise SystemExit("Explicit biometric consent is required for VIDEO_KYC enrollment.")

    with tempfile.TemporaryDirectory(prefix="phaseguard_video_kyc_") as temp_dir:
        clips = extract_phrase_clips(
            args.video,
            Path(temp_dir),
            args.start_sec,
            args.end_sec,
            args.phrase_count,
        )
        response = submit_enrollment(
            args.api_base_url,
            args.user_id,
            clips,
            consent_confirmed=True,
        )

    print(response.status_code)
    print(response.text)
    response.raise_for_status()


if __name__ == "__main__":
    main()
