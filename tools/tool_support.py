"""Runtime helpers shared by the CLI and the Tkinter GUI.

Everything here must work both from source (``tools/`` in the repo) and inside
a PyInstaller bundle, where ``sys._MEIPASS`` holds the code and the exe sits in
its own directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

APP_NAME = "MMD节奏VMD"
MODEL_FILENAME = "mms_fa.onnx"

# Meta MMS forced aligner (CC-BY-NC 4.0), republished as ONNX.
MODEL_SHA256 = "e8bad67fd3533b3d3c145b0ca31bb15383945c13384dd8975baaa7b73f7b61ac"
MODEL_SIZE = 1262421764
_MODEL_FILE = "04ac86b67129634da93aea76e0147ef3.onnx"
MODEL_URLS = (
    f"https://hf-mirror.com/deskpai/ctc_forced_aligner/resolve/main/{_MODEL_FILE}",
    f"https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/{_MODEL_FILE}",
)

CHUNK = 1 << 20


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Directory that contains the Python code."""

    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def app_dir() -> Path:
    """Directory that contains the exe (or the repo checkout)."""

    if frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Portable data directory: next to the exe when writable, else LOCALAPPDATA.

    A green build keeps ``data/`` beside the executable so the whole folder can
    be zipped, moved or deleted without leaving anything behind.
    """

    override = os.environ.get("MMD_DATA")
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    candidates = [app_dir() / "data"]
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        candidates.append(Path(base) / APP_NAME)
    candidates.append(Path.home() / APP_NAME)
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    return candidates[-1]


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(values: dict) -> None:
    try:
        settings_path().write_text(
            json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def model_candidates() -> list[Path]:
    """Places a model may already live, most preferred first."""

    candidates = [models_dir() / MODEL_FILENAME, app_dir() / "models" / MODEL_FILENAME]
    candidates.append(app_dir() / "build-test" / "models" / MODEL_FILENAME)
    return candidates


def model_path() -> Path | None:
    override = os.environ.get("MMD_MODEL")
    if override:
        path = Path(override)
        if path.is_file():
            return path
    for candidate in model_candidates():
        if candidate.is_file():
            return candidate
    return None


def default_model_path() -> Path:
    """Where a downloaded model will be stored."""

    return models_dir() / MODEL_FILENAME


def ffmpeg_path() -> str | None:
    override = os.environ.get("MMD_FFMPEG")
    if override and Path(override).is_file():
        return override
    for candidate in (
        app_dir() / "ffmpeg.exe",
        bundle_dir() / "ffmpeg.exe",
        app_dir() / "bin" / "ffmpeg.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    return shutil.which("ffmpeg")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == MODEL_SIZE and _sha256(path) == MODEL_SHA256


def download_model(progress=None, cancel=None) -> Path:
    """Download the aligner model, resuming a partial file when possible.

    ``progress(done_bytes, total_bytes, message)`` is called periodically and
    ``cancel()`` may return True to abort.
    """

    target = default_model_path()
    partial = target.with_suffix(".part")
    if target.is_file() and target.stat().st_size == MODEL_SIZE:
        if verify_model(target):
            return target
        target.unlink(missing_ok=True)

    last_error: Exception | None = None
    for url in MODEL_URLS:
        try:
            _download(url, partial, progress, cancel)
            if not verify_model(partial):
                partial.unlink(missing_ok=True)
                raise RuntimeError("下载完成但校验失败（文件可能不完整）")
            partial.replace(target)
            return target
        except Exception as error:  # noqa: BLE001 - try the next mirror
            last_error = error
            if cancel is not None and cancel():
                raise RuntimeError("已取消下载") from error
    raise RuntimeError(f"模型下载失败：{last_error}")


def _download(url: str, partial: Path, progress, cancel) -> None:
    existing = partial.stat().st_size if partial.is_file() else 0
    if existing >= MODEL_SIZE:
        partial.unlink(missing_ok=True)
        existing = 0
    headers = {"User-Agent": "MMD-rhythm-vmd/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0) + existing
        if response.status == 200 and existing:
            # Server ignored the range request: start over.
            existing = 0
            total = int(response.headers.get("Content-Length") or MODEL_SIZE)
            partial.unlink(missing_ok=True)
        mode = "ab" if existing else "wb"
        done = existing
        with partial.open(mode) as handle:
            while True:
                if cancel is not None and cancel():
                    raise RuntimeError("已取消下载")
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if progress is not None:
                    progress(done, total or MODEL_SIZE, url)
    if partial.stat().st_size != MODEL_SIZE:
        raise RuntimeError("文件大小不正确")


__all__ = [
    "APP_NAME",
    "MODEL_SIZE",
    "MODEL_SHA256",
    "app_dir",
    "bundle_dir",
    "data_dir",
    "default_model_path",
    "download_model",
    "ffmpeg_path",
    "frozen",
    "load_settings",
    "model_path",
    "save_settings",
    "settings_path",
    "verify_model",
]
