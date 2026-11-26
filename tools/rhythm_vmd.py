"""Lyrics -> rhythm matching -> MMD VMD morph animation, outside Blender.

The pipeline is deliberately split from the GUI so both the command line and
the Tkinter front-end run exactly the same code:

    text/lyrics file
      -> line splitting (punctuation aware)
      -> CTC forced alignment against the audio (optional, needs onnxruntime)
      -> G2P + viseme events (the same timeline the Blender add-on uses)
      -> per-frame morph curves -> VMD

Every run also writes an editable ``.lrc`` next to the output.  Edit that file
(line text or timings) and re-run with ``--lyrics <file>.lrc`` to regenerate the
VMD without running the aligner again.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import sys
import wave

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from mmd_mouth.core.schema import WordSegment  # noqa: E402
from mmd_mouth.core.timeline import evaluate_viseme_channels  # noqa: E402
from mmd_mouth.recognition import lyrics as lyrics_module  # noqa: E402

MORPH_BY_CHANNEL = {"A": "あ", "I": "い", "U": "う", "E": "え", "O": "お"}
CHANNELS = tuple(MORPH_BY_CHANNEL)
# Models commonly ship あ２/あ３ style variants (usually a bigger or smaller
# version of the same mouth shape).  Variant 1 is the plain name.
MORPH_VARIANT_SUFFIX = {1: "", 2: "２", 3: "３"}
DEFAULT_VARIANT = 1


def normalize_variant(value) -> int:
    """Coerce a variant selector to 1..3, falling back to 1."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return DEFAULT_VARIANT
    return min(3, max(1, number))


def channel_morph_name(channel: str, variant: int = DEFAULT_VARIANT) -> str:
    """Return the morph name for one channel at the requested variant."""

    return MORPH_BY_CHANNEL[channel] + MORPH_VARIANT_SUFFIX[normalize_variant(variant)]
def _resolve_default_model() -> Path:
    """Find the aligner model: portable data dir, then the dev checkout."""

    try:
        import tool_support  # noqa: PLC0415 - optional helper

        found = tool_support.model_path()
        if found is not None:
            return found
        # Nothing on disk yet: point at the place a first-run download will use.
        return tool_support.default_model_path()
    except Exception:  # noqa: BLE001
        pass
    return REPO_ROOT / "build-test" / "models" / "mms_fa.onnx"


DEFAULT_MODEL = _resolve_default_model()
DEFAULT_BLENDER = ""  # legacy; the tool decodes with ffmpeg now
DEFAULT_FPS = 30.0
DEFAULT_SCALE = 1.0
DEFAULT_MIN = 0.0
DEFAULT_MAX = 1.0


def clamp_channel(value: float, *, scale: float, low: float, high: float) -> float:
    """Scale then clamp one morph weight into the allowed range."""

    scaled = float(value) * float(scale)
    low = max(0.0, float(low))
    high = min(1.0, float(high))
    if high < low:
        low, high = high, low
    return max(low, min(high, scaled))


@dataclass
class LyricRow:
    """One editable lyric line."""

    start: float
    end: float
    text: str
    edited: bool = False
    spans: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def split_text(text: str) -> list[str]:
    """Split lyric text into lines, falling back to punctuation."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    body = lines[0] if lines else ""
    parts = [part.strip() for part in re.split(r"[。！？；!?;]+", body)]
    parts = [part for part in parts if part]
    if len(parts) <= 1 and len(body) > 24:
        parts = [chunk.strip() for chunk in re.split(r"[，,、]+", body) if chunk.strip()]
    return parts or ([body] if body else [])


def prepare_audio(audio: str, work_dir: Path):
    """Return a 16 kHz mono WAV path and its duration in seconds."""

    from audio_decode import ensure_wav

    work_dir.mkdir(parents=True, exist_ok=True)
    wav = ensure_wav(audio, work_dir)
    with wave.open(str(wav)) as handle:
        duration = handle.getnframes() / handle.getframerate()
    return wav, duration


def align_text(
    wav: Path,
    texts: list[str],
    *,
    model: Path = DEFAULT_MODEL,
    duration: float,
    progress=None,
):
    """Run CTC forced alignment and return per-line character spans."""

    import onnxruntime

    import ctc_align

    if progress:
        progress("加载 CTC 模型…")
    session = onnxruntime.InferenceSession(str(model))
    if progress:
        progress("读取音频…")
    waveform = ctc_align.load_wav(str(wav))
    cleaned = [[char for char in text if char.strip()] for text in texts]
    flat = [char for group in cleaned for char in group]
    if progress:
        progress(f"对齐 {len(flat)} 个字…")
    spans = ctc_align.align_characters(session, waveform, flat)
    spans = ctc_align.fill_gaps(spans, duration)
    spans = ctc_align.compact_spans(spans)
    if len(spans) != len(flat):
        raise RuntimeError("alignment returned an unexpected character count")

    grouped: list[list[tuple[str, float, float]]] = []
    cursor = 0
    for group in cleaned:
        chunk = []
        for char in group:
            start, end = spans[cursor]
            chunk.append((char, start, end))
            cursor += 1
        grouped.append(chunk)
    return grouped


def rows_from_alignment(texts: list[str], spans) -> list[LyricRow]:
    """Build editable rows from aligned character spans."""

    rows: list[LyricRow] = []
    for text, group in zip(texts, spans):
        if not group:
            continue
        rows.append(
            LyricRow(
                start=group[0][1],
                end=group[-1][2],
                text=text,
                spans=list(group),
            )
        )
    return rows


def uniform_rows(texts: list[str], start: float, end: float) -> list[LyricRow]:
    """Lay lines out without audio analysis; the manual-editing baseline."""

    document = lyrics_module.distribute_lines(texts, start, end)
    return [
        LyricRow(start=line.start_sec, end=line.end_sec, text=line.text)
        for line in document
    ]


def _group_spans(texts: list[str], spans) -> list[list[tuple[str, float, float]]]:
    grouped: list[list[tuple[str, float, float]]] = []
    cursor = 0
    for text in texts:
        characters = [char for char in text if char.strip()]
        chunk = []
        for char in characters:
            start, end = spans[cursor]
            chunk.append((char, start, end))
            cursor += 1
        grouped.append(chunk)
    return grouped


def _character_units(text: str, default_language: str) -> list[tuple[str, str]]:
    """Split a line into (character, language) pairs using the shared tokenizer."""

    units: list[tuple[str, str]] = []
    for token, language in lyrics_module.tokenize_line(
        text, default_language_code=default_language
    ):
        for character in token:
            units.append((character, language or default_language))
    return units


def _group_spans_units(per_line, spans) -> list[list[tuple[str, float, float]]]:
    grouped: list[list[tuple[str, float, float]]] = []
    cursor = 0
    for units in per_line:
        chunk = []
        for character, _language in units:
            start, end = spans[cursor]
            cursor += 1
            chunk.append((character, start, end))
        grouped.append(chunk)
    return grouped


def align_rows(
    wav: Path | str,
    texts: list[str],
    *,
    model: Path = DEFAULT_MODEL,
    duration: float,
    refine: bool = True,
    pad: float = 0.25,
    language: str = "zh-CN",
    progress=None,
) -> list[LyricRow]:
    """Full alignment: one emissions pass, then per-line refinement.

    ``language`` is the default for characters whose language cannot be
    detected from the text (Latin runs are always treated as English, kana as
    Japanese).
    """

    import onnxruntime

    import ctc_align

    if progress:
        progress("加载 CTC 模型…")
    session = onnxruntime.InferenceSession(str(model))
    if progress:
        progress("读取音频…")
    waveform = ctc_align.load_wav(str(wav))
    if progress:
        progress("计算声学特征…")
    emissions, stride_ms = ctc_align.compute_emissions(session, waveform)
    frame_seconds = stride_ms / 1000.0

    per_line = [_character_units(text, language) for text in texts]
    flat = [character for units in per_line for character, _lang in units]
    languages = [lang for units in per_line for _character, lang in units]
    if progress:
        progress(f"整首对齐 {len(flat)} 个字…")
    spans = ctc_align.align_with_emissions(
        emissions, flat, frame_seconds, 0.0, languages
    )
    spans = ctc_align.fill_gaps(spans, duration)
    spans = ctc_align.compact_spans(spans)
    rows = rows_from_alignment(texts, _group_spans_units(per_line, spans))

    if refine:
        if progress:
            progress("逐句精对齐…")
        _refine_rows(emissions, frame_seconds, rows, duration, pad, language)

    if progress:
        progress("吸附句尾到静音点…")
    trim_trailing(rows, waveform)
    return rows


def _refine_rows(
    emissions,
    frame_seconds: float,
    rows: list[LyricRow],
    duration: float,
    pad: float,
    language: str = "",
) -> None:
    """Re-align every line inside its own window, in place."""

    import ctc_align

    total_frames = len(emissions)
    for index, row in enumerate(rows):
        units = _character_units(row.text, language)
        characters = [character for character, _lang in units]
        languages = [lang for _character, lang in units]
        if not characters:
            continue
        low = max(0.0, row.start - pad)
        high = min(duration, row.end + pad)
        if index > 0:
            low = max(low, (rows[index - 1].end + row.start) / 2.0)
        if index + 1 < len(rows):
            high = min(high, (row.end + rows[index + 1].start) / 2.0)
        first = max(0, int(low / frame_seconds))
        last = min(total_frames, int(high / frame_seconds) + 1)
        if last - first < len(characters):
            continue
        spans = ctc_align.align_with_emissions(
            emissions[first:last],
            characters,
            frame_seconds,
            first * frame_seconds,
            languages,
        )
        spans = ctc_align.fill_gaps(spans, high)
        spans = ctc_align.compact_spans(spans)
        if not spans or spans[0] is None:
            continue
        row.spans = [
            (char, span[0], span[1])
            for char, span in zip(characters, spans)
            if span is not None
        ]
        if row.spans:
            row.start = row.spans[0][1]
            row.end = row.spans[-1][2]
            row.edited = False


def trim_trailing(rows: list[LyricRow], waveform, *, hop_sec: float = 0.02, search: float = 0.45) -> None:
    """Pull each line end back to a quiet point before the next line starts.

    Even after refinement the last character can hold a pause.  The quietest
    frame inside the trailing window is the most likely phrase boundary, so the
    row (and its last character span) is trimmed there.
    """

    import numpy as np

    if not rows or waveform is None or len(waveform) == 0:
        return
    hop = max(1, int(round(hop_sec * 16000)))
    rms = np.array(
        [
            float(np.sqrt(np.mean(waveform[i : i + hop] ** 2)))
            for i in range(0, len(waveform) - hop, hop)
        ]
    )
    if rms.size == 0:
        return

    def energy_at(time_sec: float) -> float:
        index = int(time_sec / hop_sec)
        if index < 0 or index >= rms.size:
            return 1.0
        return float(rms[index])

    for index, row in enumerate(rows[:-1]):
        following = rows[index + 1]
        low = max(row.start + 0.15, row.end - search)
        high = min(row.end + 0.05, following.start)
        if high <= low:
            continue
        first = int(low / hop_sec)
        last = int(high / hop_sec) + 1
        window = rms[first:last]
        if window.size == 0:
            continue
        best = first + int(window.argmin())
        cut = best * hop_sec
        if cut >= row.end - 0.02:
            continue
        # Only accept a dip that is clearly quieter than the line average.
        line_first = int(row.start / hop_sec)
        line_energy = float(rms[line_first:first].mean()) if first > line_first else 0.0
        if line_energy > 0.0 and window.min() > line_energy * 0.9:
            continue
        row.end = cut
        if row.spans:
            char, start, end = row.spans[-1]
            if end > cut:
                row.spans[-1] = (char, start, cut)


def refine_rows(
    wav: Path | str,
    rows: list[LyricRow],
    *,
    model: Path = DEFAULT_MODEL,
    duration: float,
    pad: float = 0.25,
    language: str = "",
    progress=None,
) -> list[LyricRow]:
    """Standalone refinement entry point (loads the model and emissions)."""

    import onnxruntime

    import ctc_align

    if not rows:
        return rows
    if progress:
        progress("逐句精对齐…")
    session = onnxruntime.InferenceSession(str(model))
    waveform = ctc_align.load_wav(str(wav))
    emissions, stride_ms = ctc_align.compute_emissions(session, waveform)
    _refine_rows(emissions, stride_ms / 1000.0, rows, duration, pad, language)
    trim_trailing(rows, waveform)
    return rows


def rows_to_words(rows: list[LyricRow], language: str) -> list[WordSegment]:
    """Turn rows into timed words, preferring aligned characters."""

    words: list[WordSegment] = []
    for row in rows:
        if row.spans and not row.edited:
            for char, start, end in row.spans:
                words.append(
                    WordSegment(
                        text=char,
                        start_sec=start,
                        end_sec=max(start, end),
                        language_code=language,
                        confidence=1.0,
                        model_id="rhythm",
                    )
                )
            continue
        characters = [char for char in row.text if char.strip()]
        if not characters:
            continue
        document = lyrics_module.LyricsDocument(
            lines=[lyrics_module.LyricLine(start_sec=row.start, end_sec=row.end, text=row.text)]
        )
        for word in lyrics_module.to_word_segments(document, default_language_code=language):
            words.append(word)
    words.sort(key=lambda item: (item.start_sec, item.end_sec))
    return words


def build_events(
    words: list[WordSegment],
    *,
    language: str,
    attack_ms: float = 70.0,
    release_ms: float = 90.0,
    blend: bool = True,
):
    """Run the shared G2P + viseme timeline."""

    return lyrics_module.run_lyrics_pipeline(
        words,
        default_language_code=language,
        attack_ms=attack_ms,
        release_ms=release_ms,
        blend=blend,
    )


def sample_curves(
    events,
    duration: float,
    *,
    fps: float = DEFAULT_FPS,
    attack_ms: float = 70.0,
    release_ms: float = 90.0,
    hold_ratio: float = 0.55,
    epsilon: float = 0.015,
    dense: bool = True,
    style: str = "CLASSIC",
    smooth_sigma: float | None = None,
    scale: float = DEFAULT_SCALE,
    limits: dict[str, tuple[float, float]] | None = None,
    variants: dict[str, int] | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """Sample the viseme timeline at ``fps`` and simplify each morph channel.

    ``dense`` writes a key on every frame.  VMD morph records carry no
    interpolation data, so a reader is free to treat sparse keys as steps;
    per-frame keys look correct no matter how the file is imported.

    ``style`` is CLASSIC (the original snappy per-phoneme shapes) or NATURAL
    (overlapping bell curves plus smoothing).

    ``scale`` multiplies every weight before ``limits`` clamps it, so the mouth
    can be made smaller without flattening the motion shape.
    """

    from mmd_mouth.core.timeline import smooth_values
    from vmd_writer import simplify

    if smooth_sigma is None:
        smooth_sigma = 1.8 if style == "NATURAL" else 0.0
    total = max(1, int(round(duration * fps)))
    curves: dict[str, list[tuple[int, float]]] = {}
    for channel in CHANNELS:
        low, high = (limits or {}).get(channel, (DEFAULT_MIN, DEFAULT_MAX))
        points: list[tuple[int, float]] = []
        for frame in range(total + 1):
            values = evaluate_viseme_channels(
                events,
                frame / fps,
                attack_ms=attack_ms,
                release_ms=release_ms,
                hold_ratio=hold_ratio,
                style=style,
            )
            points.append((frame, float(values[channel])))
        if smooth_sigma > 0.0:
            blurred = smooth_values([value for _frame, value in points], smooth_sigma)
            points = [(frame, value) for (frame, _), value in zip(points, blurred)]
        points = [
            (frame, round(clamp_channel(value, scale=scale, low=low, high=high), 4))
            for frame, value in points
        ]
        morph = channel_morph_name(channel, (variants or {}).get(channel, DEFAULT_VARIANT))
        curves[morph] = points if dense else simplify(points, epsilon)
    return curves


def slice_wav(source: Path | str, start: float, end: float, target: Path | str) -> Path:
    """Copy ``[start, end]`` of a PCM WAV into ``target`` (for previewing)."""

    target = Path(target)
    with wave.open(str(source)) as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        total = reader.getnframes()
        first = max(0, int(round(start * rate)))
        last = min(total, int(round(end * rate)))
        if last <= first:
            last = min(total, first + int(rate * 0.2))
        reader.setpos(first)
        data = reader.readframes(last - first)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(data)
    return target


def write_lrc(rows: list[LyricRow], path: Path) -> None:
    lines = []
    for row in rows:
        minutes = int(row.start // 60)
        seconds = row.start % 60
        lines.append(f"[{minutes:02d}:{seconds:05.2f}]{row.text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_align_json(rows: list[LyricRow], path: Path, duration: float, language: str) -> None:
    payload = {
        "version": 1,
        "language": language,
        "audio_duration_sec": round(duration, 3),
        "lines": [
            {
                "lang": language,
                "start": round(row.start, 3),
                "end": round(row.end, 3),
                "text": row.text,
                "words": [
                    {"text": char, "start": round(start, 3), "end": round(end, 3), "lang": language}
                    for char, start, end in row.spans
                ]
                or [
                    {
                        "text": char,
                        "start": round(row.start + index * row.duration / max(1, len(row.text)), 3),
                        "end": round(
                            row.start + (index + 1) * row.duration / max(1, len(row.text)), 3
                        ),
                        "lang": language,
                    }
                    for index, char in enumerate([c for c in row.text if c.strip()])
                ],
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate(
    *,
    audio: str,
    lyrics_path: str,
    out_dir: str,
    name: str = "",
    language: str = "zh-CN",
    model: str = "",
    fps: float = DEFAULT_FPS,
    attack_ms: float = 70.0,
    release_ms: float = 90.0,
    no_align: bool = False,
    refine: bool = True,
    blend: bool = True,
    dense: bool = True,
    style: str = "CLASSIC",
    scale: float = DEFAULT_SCALE,
    limits: dict[str, tuple[float, float]] | None = None,
    variants: dict[str, int] | None = None,
    progress=None,
) -> dict:
    """Full pipeline; returns the artefact paths and statistics."""

    from vmd_writer import write_vmd

    out_root = Path(out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    stem = name or Path(lyrics_path).stem
    lyrics_file = Path(lyrics_path).expanduser()

    wav, duration = prepare_audio(audio, out_root / "_work")
    if progress:
        progress(f"音频 {duration:.1f}s")

    suffix = lyrics_file.suffix.lower()
    rows: list[LyricRow]
    if suffix in {".lrc", ".qrc", ".json", ".align"} and not no_align:
        document = lyrics_module.load_lyrics(
            str(lyrics_file),
            default_language_code=language,
            audio_duration_sec=duration,
        )
        rows = [
            LyricRow(start=line.start_sec, end=line.end_sec, text=line.text)
            for line in document.lines
        ]
        if progress:
            progress(f"读取时间轴 {len(rows)} 行")
    else:
        texts = split_text(lyrics_file.read_text(encoding="utf-8-sig"))
        if not texts:
            raise RuntimeError("歌词文件里没有可用的行")
        if progress:
            progress(f"自动分割 {len(texts)} 行")
        if no_align:
            rows = uniform_rows(texts, 0.0, duration)
        else:
            rows = align_rows(
                wav,
                texts,
                model=Path(model) if model else DEFAULT_MODEL,
                duration=duration,
                refine=refine,
                language=language,
                progress=progress,
            )
        if progress:
            progress(f"对齐完成：{len(rows)} 行")

    words = rows_to_words(rows, language)
    if not words:
        raise RuntimeError("没有可用歌词")
    if progress:
        progress(f"音素事件 {len(words)} 词")
    document = build_events(
        words,
        language=language,
        attack_ms=attack_ms,
        release_ms=release_ms,
        blend=blend,
    )
    curves = sample_curves(
        document.events,
        duration,
        fps=fps,
        attack_ms=attack_ms,
        release_ms=release_ms,
        dense=dense,
        style=style,
        scale=scale,
        limits=limits,
        variants=variants,
    )

    vmd_path = out_root / f"{stem}.vmd"
    keys = write_vmd(vmd_path, curves)
    lrc_path = out_root / f"{stem}.lrc"
    write_lrc(rows, lrc_path)
    json_path = out_root / f"{stem}.align.json"
    write_align_json(rows, json_path, duration, language)
    if progress:
        progress(f"VMD 已写出（{keys} 个关键帧）")

    return {
        "vmd": vmd_path,
        "lrc": lrc_path,
        "align_json": json_path,
        "lines": len(rows),
        "words": len(words),
        "events": len(document.events),
        "morph_keys": keys,
        "duration": duration,
        "curves": {name: len(points) for name, points in curves.items()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--out", default="vmd_out")
    parser.add_argument("--name", default="")
    parser.add_argument("--language", default="zh-CN")
    parser.add_argument("--model", default="")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--attack-ms", type=float, default=70.0)
    parser.add_argument("--release-ms", type=float, default=90.0)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="skip the per-line re-alignment pass",
    )
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="use single-vowel events instead of per-syllable blends",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="drop redundant keys (smaller VMD, readers may treat gaps as steps)",
    )
    parser.add_argument(
        "--style",
        choices=("classic", "natural"),
        default="classic",
        help="classic = original snappy shapes, natural = bell curves",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="multiply every morph weight before clamping (0.2-1.5)",
    )
    parser.add_argument("--min", type=float, default=DEFAULT_MIN, help="lower clamp")
    parser.add_argument("--max", type=float, default=DEFAULT_MAX, help="upper clamp")
    parser.add_argument(
        "--limits",
        default="",
        help='per-channel JSON, e.g. {"あ": [0.0, 0.8], "い": [0.0, 1.0]}',
    )
    parser.add_argument(
        "--variants",
        default="",
        help='per-channel morph variant 1/2/3, e.g. "A=2,U=2" (default all 1)',
    )
    args = parser.parse_args(argv)

    limits: dict[str, tuple[float, float]] = {}
    if args.limits:
        data = json.loads(Path(args.limits).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise SystemExit("limits JSON must be an object")
        morph_to_channel = {morph: channel for channel, morph in MORPH_BY_CHANNEL.items()}
        for name, pair in data.items():
            key = str(name)
            channel = key if key in CHANNELS else morph_to_channel.get(key)
            if channel is None:
                raise SystemExit(f"unknown morph in limits JSON: {key}")
            limits[channel] = (float(pair[0]), float(pair[1]))
    for channel in CHANNELS:
        limits.setdefault(channel, (args.min, args.max))

    variants: dict[str, int] = {}
    if args.variants:
        for part in args.variants.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"variants must look like A=2,U=2 (got {part!r})")
            key, value = part.split("=", 1)
            channel = key.strip().upper()
            if channel not in CHANNELS:
                raise SystemExit(f"unknown channel in --variants: {key}")
            variants[channel] = normalize_variant(value)

    stats = generate(
        audio=args.audio,
        lyrics_path=args.lyrics,
        out_dir=args.out,
        name=args.name,
        language=args.language,
        model=args.model,
        fps=args.fps,
        attack_ms=args.attack_ms,
        release_ms=args.release_ms,
        no_align=args.no_align,
        refine=not args.no_refine,
        blend=not args.no_blend,
        dense=not args.sparse,
        style="NATURAL" if args.style == "natural" else "CLASSIC",
        scale=args.scale,
        limits=limits,
        variants=variants,
        progress=lambda message: print("·", message, flush=True),
    )
    print("VMD:", stats["vmd"])
    print("LRC:", stats["lrc"])
    print("ALIGN:", stats["align_json"])
    print("lines", stats["lines"], "keys", stats["morph_keys"], "curves", stats["curves"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
