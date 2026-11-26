"""Decode any audio file into the mono 16 kHz PCM WAV the aligner needs.

ffmpeg is the only dependency: the standalone build ships ``ffmpeg.exe`` next
to the executable, and development runs use the copy from ``imageio-ffmpeg``.
Blender is no longer required.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

SAMPLE_RATE = 16000


def _ffmpeg() -> str:
    try:
        import tool_support

        found = tool_support.ffmpeg_path()
        if found:
            return found
    except Exception:  # noqa: BLE001 - optional helper
        pass
    import shutil

    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError(
            "找不到 ffmpeg。请把它放在程序目录，或安装 imageio-ffmpeg。"
        )
    return found


def ensure_wav(audio: Path | str, work: Path | str, sample_rate: int = SAMPLE_RATE) -> Path:
    """Return a mono PCM WAV at ``sample_rate`` for ``audio``."""

    audio = Path(audio)
    if not audio.is_file():
        raise FileNotFoundError(f"音频不存在：{audio}")

    if audio.suffix.lower() == ".wav":
        with wave.open(str(audio)) as handle:
            if handle.getframerate() == sample_rate and handle.getnchannels() == 1:
                return audio

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    target = work / "decoded.wav"
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(audio),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


__all__ = ["SAMPLE_RATE", "ensure_wav"]
