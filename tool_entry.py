"""Entry point for the standalone (PyInstaller) build of the rhythm VMD tool.

    MMDRhythmVMD.exe              launch the GUI
    MMDRhythmVMD.exe --selftest   print environment facts and exit
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> Path:
    """Make the bundled code importable, frozen or not."""

    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    for candidate in (base / "tools", base):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)
    return base


def _selftest() -> int:
    import os
    import platform

    lines: list[str] = []

    def say(message: str) -> None:
        print(message)
        lines.append(str(message))

    say(f"python      {platform.python_version()}")
    say(f"frozen      {bool(getattr(sys, 'frozen', False))}")
    say(f"bundle      {_bootstrap()}")

    code = 0
    try:
        import tool_support

        say(f"app_dir     {tool_support.app_dir()}")
        say(f"data_dir    {tool_support.data_dir()}")
        say(f"model       {tool_support.model_path()}")
        say(f"ffmpeg      {tool_support.ffmpeg_path()}")
        for module in ("numpy", "onnxruntime", "pypinyin", "cmudict", "pyopenjtalk"):
            imported = __import__(module)
            version = getattr(imported, "__version__", "?")
            say(f"{module:11s} {version}")
        import rhythm_vmd

        say(f"DEFAULT_MODEL {rhythm_vmd.DEFAULT_MODEL}")
        from mmd_mouth.language.g2p import pronunciation_tokens

        for text, language in (("君の声", "ja-JP"), ("baby", "en-US"), ("爱", "zh-CN")):
            phones = ",".join(token.source for token in pronunciation_tokens(text, language))
            say(f"g2p {language} {text} -> {phones}")
        from mmd_mouth.language.romanize import romanize_letters

        say("romanize ja -> " + "".join(romanize_letters("君の声が", "ja-JP")))
    except Exception as error:  # noqa: BLE001
        say(f"FAILED: {error}")
        code = 1
    finally:
        try:
            target = Path(os.environ.get("TEMP", ".")) / "mmd_rhythm_selftest.txt"
            target.write_text("\n".join(lines), encoding="utf-8")
            print(f"written to {target}")
        except OSError:
            pass
    return code


def _selftest_align(audio: str) -> int:
    """Decode an audio file and align two lines, exercising the whole stack."""

    import time
    from pathlib import Path

    import rhythm_vmd as core

    started = time.time()
    print("model     ", core.DEFAULT_MODEL)
    wav, duration = core.prepare_audio(audio, Path("data") / "_selftest")
    print(f"decoded    {wav.name}  {duration:.2f}s  ({time.time() - started:.1f}s)")
    rows = core.align_rows(
        wav,
        ["我爱你的 baby", "君の声が"],
        duration=duration,
        refine=False,
    )
    for row in rows:
        print(f"line       {row.start:7.2f} - {row.end:7.2f}  {row.text}")
    print(f"aligned in {time.time() - started:.1f}s")
    return 0


def main() -> int:
    _bootstrap()
    if "--selftest" in sys.argv:
        return _selftest()
    if "--selftest-align" in sys.argv:
        index = sys.argv.index("--selftest-align")
        audio = sys.argv[index + 1]
        return _selftest_align(audio)
    import rhythm_vmd_gui

    return rhythm_vmd_gui.main()


if __name__ == "__main__":
    raise SystemExit(main())
