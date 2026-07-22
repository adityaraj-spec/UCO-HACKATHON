from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly


@dataclass(frozen=True)
class ChannelCondition:
    name: str
    snr_db: float | None


CONDITIONS: tuple[ChannelCondition, ...] = (
    ChannelCondition("clean", None),
    ChannelCondition("telephone_mild", 20.0),
    ChannelCondition("telephone_moderate", 10.0),
    ChannelCondition("telephone_severe", 3.0),
)


def read_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio, sr


def write_audio(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr)


def simulate_file(
    source_path: Path,
    destination_path: Path,
    condition: str,
    noise_files: list[Path],
    seed: int,
    target_sr: int = 16000,
) -> None:
    audio, sr = read_audio(source_path, target_sr=target_sr)
    simulated = simulate_audio(audio, sr, condition, noise_files, seed)
    write_audio(destination_path, simulated, sr)


def simulate_audio(
    audio: np.ndarray,
    sr: int,
    condition: str,
    noise_files: list[Path],
    seed: int,
) -> np.ndarray:
    if condition == "clean":
        return audio.astype(np.float32)

    selected = next((item for item in CONDITIONS if item.name == condition), None)
    if selected is None:
        raise ValueError(f"Unknown channel condition: {condition}")
    if selected.snr_db is None:
        return audio.astype(np.float32)

    filtered = _bandpass(audio, sr, low_hz=300.0, high_hz=3400.0)
    coded = _telephone_codec_roundtrip(filtered, sr)
    noisy = _add_noise(coded, selected.snr_db, noise_files, seed, sr)

    peak = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    if peak > 0:
        noisy = 0.98 * noisy / peak
    return noisy.astype(np.float32)


def _bandpass(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = sr / 2.0
    high = min(high_hz / nyquist, 0.999)
    low = max(low_hz / nyquist, 0.001)
    b, a = butter(6, [low, high], btype="band")
    return lfilter(b, a, audio).astype(np.float32)


def _telephone_codec_roundtrip(audio: np.ndarray, sr: int) -> np.ndarray:
    down = resample_poly(audio, 8000, sr)
    quantized = np.round(np.clip(down, -1.0, 1.0) * 32767.0) / 32767.0
    return resample_poly(quantized, sr, 8000).astype(np.float32)


def _add_noise(
    audio: np.ndarray,
    snr_db: float,
    noise_files: list[Path],
    seed: int,
    sr: int,
) -> np.ndarray:
    rng = Random(seed)
    if noise_files:
        noise, _ = read_audio(rng.choice(noise_files), target_sr=sr)
    else:
        noise = np.asarray([rng.uniform(-1.0, 1.0) for _ in range(max(1, len(audio)))], dtype=np.float32)

    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    start_max = max(0, len(noise) - len(audio))
    start = rng.randint(0, start_max) if start_max else 0
    noise = noise[start : start + len(audio)].astype(np.float32)

    speech_power = float(np.mean(audio**2))
    noise_power = float(np.mean(noise**2))
    if speech_power <= 0.0 or noise_power <= 0.0:
        return audio.astype(np.float32)

    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return (audio + noise).astype(np.float32)
