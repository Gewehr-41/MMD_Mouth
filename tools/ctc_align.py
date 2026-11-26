"""CTC forced alignment with the MMS-FA ONNX model.

Self-contained replacement for ``ctc_forced_aligner``'s aligner: the PyPI
package ships its Viterbi step as a compiled C++ extension, which needs MSVC to
build.  The model itself is plain ONNX and the algorithm is standard CTC
alignment, so it is implemented here in numpy instead.

Input text is romanized with pypinyin (already a worker dependency), which maps
Mandarin characters onto the Latin letter vocabulary the model expects.
"""

from __future__ import annotations

from pathlib import Path
import math
import wave

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 20.0

# MMS-FA token order, matching the ONNX logits (pad/</s>/unk are placeholders
# kept so letter indices line up with the model's real ordering).
VOCAB = {
    "<blank>": 0,
    "<pad>": 1,
    "</s>": 2,
    "<unk>": 3,
    "a": 4, "i": 5, "e": 6, "n": 7, "o": 8, "u": 9, "t": 10, "s": 11, "r": 12,
    "m": 13, "k": 14, "l": 15, "d": 16, "g": 17, "h": 18, "y": 19, "b": 20,
    "p": 21, "w": 22, "c": 23, "v": 24, "j": 25, "z": 26, "f": 27, "'": 28,
    "q": 29, "x": 30,
}
BLANK = 0


def load_wav(path: Path) -> np.ndarray:
    """Read a 16 kHz mono PCM WAV as float32 in [-1, 1]."""

    with wave.open(str(path)) as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise ValueError("CTC alignment needs a 16 kHz mono WAV")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def compute_emissions(
    session,
    waveform: np.ndarray,
    *,
    window_sec: float = 30.0,
    context_sec: float = 2.0,
    batch_size: int = 4,
):
    """Run the ONNX model in overlapping windows and return log-probabilities."""

    context = int(context_sec * SAMPLE_RATE)
    window = int(window_sec * SAMPLE_RATE)
    extension = math.ceil(len(waveform) / window) * window - len(waveform)
    padded = np.pad(waveform, (context, context + extension), mode="constant")
    count = (len(padded) - 2 * context) // window
    windows = np.stack(
        [padded[i * window : i * window + window + 2 * context] for i in range(count)]
    )

    outputs = []
    for start in range(0, count, batch_size):
        batch = windows[start : start + batch_size].astype(np.float32)
        outputs.append(session.run(["logits"], {"input_values": batch})[0])
    emissions = np.concatenate(outputs, axis=0)

    context_frames = int(context_sec * 1000.0 / FRAME_MS)
    if context_frames > 0:
        emissions = emissions[:, context_frames : -context_frames + 1, :]
    emissions = emissions.reshape(-1, emissions.shape[-1])

    extra = int(extension / SAMPLE_RATE * 1000.0 / FRAME_MS)
    if extra > 0:
        emissions = emissions[:-extra, :]

    peak = emissions.max(axis=-1, keepdims=True)
    emissions = emissions - peak - np.log(np.exp(emissions - peak).sum(axis=-1, keepdims=True))
    stride_ms = len(waveform) * 1000.0 / emissions.shape[0] / SAMPLE_RATE
    return emissions.astype(np.float32), stride_ms


def viterbi(log_probs: np.ndarray, targets: list[int]):
    """Standard CTC forced alignment returning a per-frame state path."""

    total, classes = log_probs.shape
    length = len(targets)
    if length == 0:
        return np.zeros(total, dtype=np.int64), np.ones(total, dtype=np.float32)

    states = np.full(2 * length + 1, BLANK, dtype=np.int64)
    states[1::2] = targets
    skip = np.zeros(2 * length + 1, dtype=bool)
    if length > 1:
        skip[3::2] = np.asarray(targets[1:]) != np.asarray(targets[:-1])

    negative = np.float32(-1e30)
    dp = np.full((total, len(states)), negative, dtype=np.float32)
    dp[0, 0] = log_probs[0, BLANK]
    if len(states) > 1:
        dp[0, 1] = log_probs[0, targets[0]]

    for frame in range(1, total):
        previous = dp[frame - 1]
        best = previous.copy()
        np.maximum(best[1:], previous[:-1], out=best[1:])
        jump = np.full(len(states), negative, dtype=np.float32)
        jump[2:] = previous[:-2]
        jump[~skip] = negative
        np.maximum(best, jump, out=best)
        dp[frame] = best + log_probs[frame, states]

    final = dp[-1]
    state = len(states) - 1
    if len(states) > 1 and final[-2] > final[-1]:
        state = len(states) - 2

    path = np.empty(total, dtype=np.int64)
    frames = np.empty(total, dtype=np.float32)
    for frame in range(total - 1, -1, -1):
        path[frame] = state
        frames[frame] = dp[frame, state]
        if frame == 0:
            break
        previous = dp[frame - 1]
        candidates = [(previous[state], state)]
        if state > 0:
            candidates.append((previous[state - 1], state - 1))
        if state > 1 and skip[state]:
            candidates.append((previous[state - 2], state - 2))
        state = max(candidates, key=lambda item: item[0])[1]
    return path, np.exp(frames - frames.max())


def _pinyin_letters(character: str, language_code: str = "") -> list[str]:
    """Romanize one piece of text for the model, honouring its language."""

    from mmd_mouth.language.romanize import romanize_letters

    return romanize_letters(character, language_code)


def _target_letters(characters: list[str], languages: list[str] | None = None):
    """Map characters to model letter ids, remembering which ids each owns."""

    letter_ids: list[int] = []
    char_ranges: list[tuple[int, int] | None] = []
    for index, character in enumerate(characters):
        language = ""
        if languages is not None and index < len(languages):
            language = languages[index] or ""
        letters = _pinyin_letters(character, language)
        ids = [VOCAB[letter] for letter in letters if letter in VOCAB]
        if not ids:
            char_ranges.append(None)
            continue
        start = len(letter_ids)
        letter_ids.extend(ids)
        char_ranges.append((start, len(letter_ids) - 1))
    return letter_ids, char_ranges


def _spans_from_path(path, char_ranges, frame_seconds: float, offset: float = 0.0):
    last_letter = max((value[1] for value in char_ranges if value), default=-1)
    letter_spans: list[list[int] | None] = [None] * (last_letter + 1)
    for frame, state in enumerate(path):
        if state <= 0:
            continue
        index = (int(state) - 1) // 2
        if index < 0 or index >= len(letter_spans):
            continue
        span = letter_spans[index]
        if span is None:
            letter_spans[index] = [frame, frame]
        else:
            span[1] = frame

    results: list[tuple[float, float] | None] = []
    for char_range in char_ranges:
        if char_range is None:
            results.append(None)
            continue
        first, last = char_range
        starts = [
            letter_spans[i][0]
            for i in range(first, last + 1)
            if i < len(letter_spans) and letter_spans[i]
        ]
        ends = [
            letter_spans[i][1]
            for i in range(first, last + 1)
            if i < len(letter_spans) and letter_spans[i]
        ]
        if not starts:
            results.append(None)
            continue
        results.append(
            (
                offset + min(starts) * frame_seconds,
                offset + (max(ends) + 1) * frame_seconds,
            )
        )
    return results


def align_with_emissions(
    emissions,
    characters: list[str],
    frame_seconds: float,
    offset: float = 0.0,
    languages: list[str] | None = None,
):
    """Align characters against already-computed emissions."""

    letter_ids, char_ranges = _target_letters(characters, languages)
    if not letter_ids:
        raise ValueError("no characters could be romanized for alignment")
    path, _scores = viterbi(emissions, letter_ids)
    return _spans_from_path(path, char_ranges, frame_seconds, offset)


def align_characters(
    session,
    waveform: np.ndarray,
    characters: list[str],
    languages: list[str] | None = None,
):
    """Return ``(start_sec, end_sec)`` per character, or None when unaligned."""

    emissions, stride_ms = compute_emissions(session, waveform)
    return align_with_emissions(
        emissions, characters, stride_ms / 1000.0, 0.0, languages
    )


def fill_gaps(results, duration: float):
    """Interpolate characters the model could not place."""

    known = [(index, value) for index, value in enumerate(results) if value]
    if not known:
        step = duration / max(1, len(results))
        return [(step * i, step * (i + 1)) for i in range(len(results))]

    filled: list[tuple[float, float]] = [value for value in results]  # type: ignore[list-item]
    first_index, first_value = known[0]
    for index in range(first_index):
        offset = (first_index - index) * 0.2
        filled[index] = (max(0.0, first_value[0] - offset), max(0.0, first_value[0] - offset + 0.15))
    for (prev_index, prev_value), (next_index, next_value) in zip(known, known[1:]):
        for index in range(prev_index + 1, next_index):
            ratio = (index - prev_index) / (next_index - prev_index)
            start = prev_value[1] + (next_value[0] - prev_value[1]) * ratio
            filled[index] = (start, start + 0.12)
    last_index, last_value = known[-1]
    for index in range(last_index + 1, len(filled)):
        start = last_value[1] + (index - last_index) * 0.2
        filled[index] = (start, start + 0.15)

    for index in range(1, len(filled)):
        if filled[index][0] < filled[index - 1][1]:
            filled[index] = (filled[index - 1][1], max(filled[index - 1][1], filled[index][1]))
    return filled


def compact_spans(spans, factor: float = 3.0):
    """Pull abnormally long character spans to their trailing edge.

    CTC alignment has to place every token somewhere.  When an instrumental
    section sits between two sung lines, the alignment often stretches the last
    character before the gap across it.  The vocal is at the *end* of such a
    span, so the start is moved to a typical syllable length before the end and
    the gap becomes silence.
    """

    durations = sorted(
        span[1] - span[0] for span in spans if span and span[1] > span[0]
    )
    if not durations:
        return spans
    median = durations[len(durations) // 2]
    if median <= 0.0:
        return spans
    limit = max(2.0, median * factor)
    typical = max(0.35, median * 1.2)
    result = []
    last_index = len(spans) - 1
    for index, span in enumerate(spans):
        if span is None:
            result.append(None)
            continue
        start, end = span
        if end - start > limit:
            if index == last_index:
                # Trailing instrumental: the vocal is at the beginning.
                end = start + typical
            else:
                # Leading gap: the vocal sits at the end of the span.
                start = max(0.0, end - typical)
        result.append((start, end))
    return result


__all__ = [
    "SAMPLE_RATE",
    "VOCAB",
    "align_characters",
    "compact_spans",
    "compute_emissions",
    "fill_gaps",
    "load_wav",
    "viterbi",
]
