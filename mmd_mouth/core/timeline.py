"""Consonant-aware viseme events and deterministic frame sampling."""

from __future__ import annotations

from math import ceil, cos, exp, isclose, pi
from typing import Dict, Iterable, Mapping, Sequence

from ..constants import (
    DEFAULT_ATTACK_MS,
    DEFAULT_HOLD_RATIO,
    DEFAULT_MOTION_STYLE,
    DEFAULT_RELEASE_MS,
)
from .schema import PhonemeSegment, VisemeEvent


VISEME_CHANNELS = ("REST", "CLOSED", "A", "I", "U", "E", "O")
VOWEL_CHANNELS = ("A", "I", "U", "E", "O")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ease_progress(progress: float, easing_mode: str) -> float:
    value = _clamp(progress)
    if easing_mode == "SMOOTHSTEP":
        return value * value * (3.0 - 2.0 * value)
    if easing_mode == "SINE":
        return 0.5 - 0.5 * cos(pi * value)
    if easing_mode == "EASE_IN":
        return value * value
    if easing_mode == "EASE_OUT":
        return 1.0 - (1.0 - value) * (1.0 - value)
    return value


def build_viseme_events(
    phonemes: Sequence[PhonemeSegment],
    *,
    attack_ms: float = DEFAULT_ATTACK_MS,
    release_ms: float = DEFAULT_RELEASE_MS,
) -> list[VisemeEvent]:
    """Convert aligned IPA phones into raw vowel and suppression events.

    ``attack_ms`` and ``release_ms`` remain accepted for worker compatibility,
    but transition windows are intentionally evaluated later. This keeps the
    recognized timeline editable and lets a new bake use different timings
    without asking Vosk to run again.
    """

    del attack_ms, release_ms

    events: list[VisemeEvent] = []
    for index, phoneme in enumerate(phonemes):
        common = {
            "confidence": phoneme.confidence,
            "source": "G2P",
            "source_index": index,
            "source_text": phoneme.source_text,
            "phoneme": phoneme.phoneme,
            "language_code": phoneme.language_code,
            "source_phoneme": phoneme.source_phoneme,
            "articulation_class": phoneme.articulation_class,
        }
        if phoneme.viseme_id == "CLOSED" and phoneme.close_strength > 0.0:
            events.append(
                VisemeEvent(
                    viseme_id="CLOSED",
                    start_sec=phoneme.start_sec,
                    end_sec=phoneme.end_sec,
                    weight=_clamp(phoneme.close_strength),
                    priority=100,
                    **common,
                )
            )
            continue
        if phoneme.viseme_id in VOWEL_CHANNELS:
            weight = 1.0 if phoneme.phoneme_type == "VOWEL" else 0.65
            events.append(
                VisemeEvent(
                    viseme_id=phoneme.viseme_id,
                    start_sec=phoneme.start_sec,
                    end_sec=phoneme.end_sec,
                    weight=weight,
                    priority=50,
                    **common,
                )
            )
        if phoneme.vowel_suppression > 0.0:
            events.append(
                VisemeEvent(
                    viseme_id="REST",
                    start_sec=phoneme.start_sec,
                    end_sec=phoneme.end_sec,
                    weight=_clamp(phoneme.vowel_suppression),
                    priority=80,
                    **common,
                )
            )
    return sorted(events, key=lambda value: (value.start_sec, -value.priority))


def _envelope_classic(
    event: VisemeEvent,
    time_sec: float,
    *,
    attack_sec: float,
    release_sec: float,
    hold_ratio: float,
    easing_mode: str,
) -> float:
    """Original per-phoneme envelope: hold plateau plus short ramps."""

    duration = event.end_sec - event.start_sec
    if duration <= 0.0:
        return event.weight if time_sec == event.start_sec else 0.0

    if easing_mode != "LINEAR" or event.viseme_id not in VOWEL_CHANNELS:
        active_start = event.start_sec - attack_sec
        active_end = event.end_sec + release_sec
        if time_sec < active_start or time_sec > active_end:
            return 0.0
        if attack_sec > 0.0 and time_sec < event.start_sec:
            progress = (time_sec - active_start) / attack_sec
            return event.weight * _ease_progress(progress, easing_mode)
        if release_sec > 0.0 and time_sec > event.end_sec:
            progress = (time_sec - event.end_sec) / release_sec
            return event.weight * (1.0 - _ease_progress(progress, easing_mode))
        return event.weight

    edge_duration = duration * (1.0 - _clamp(hold_ratio))
    requested_edges = attack_sec + release_sec
    if requested_edges > 0.0 and requested_edges > edge_duration:
        scale = edge_duration / requested_edges
        actual_attack = attack_sec * scale
        actual_release = release_sec * scale
    else:
        actual_attack = min(attack_sec, edge_duration)
        actual_release = min(release_sec, max(0.0, edge_duration - actual_attack))

    if time_sec < event.start_sec or time_sec > event.end_sec:
        return 0.0
    if actual_attack > 0.0 and time_sec < event.start_sec + actual_attack:
        progress = (time_sec - event.start_sec) / actual_attack
        return event.weight * _ease_progress(progress, easing_mode)
    if actual_release > 0.0 and time_sec > event.end_sec - actual_release:
        progress = (event.end_sec - time_sec) / actual_release
        return event.weight * _ease_progress(progress, easing_mode)
    return event.weight


def _envelope(
    event: VisemeEvent,
    time_sec: float,
    *,
    attack_sec: float,
    release_sec: float,
    hold_ratio: float,
    easing_mode: str,
    style: str = "CLASSIC",
) -> float:
    if style != "NATURAL":
        return _envelope_classic(
            event,
            time_sec,
            attack_sec=attack_sec,
            release_sec=release_sec,
            hold_ratio=hold_ratio,
            easing_mode=easing_mode,
        )

    duration = event.end_sec - event.start_sec
    if duration <= 0.0:
        return event.weight if time_sec == event.start_sec else 0.0

    if event.viseme_id not in VOWEL_CHANNELS:
        # Closure/silence are short articulations; a long vowel-style ramp
        # would keep the mouth shut well after the consonant is over.  Their
        # ramps live outside the event so they can overlap the vowels.
        actual_attack = min(attack_sec, 0.05)
        actual_release = min(release_sec, 0.05)
        active_start = event.start_sec - actual_attack
        active_end = event.end_sec + actual_release
        if time_sec < active_start or time_sec > active_end:
            return 0.0
        if actual_attack > 0.0 and time_sec < event.start_sec:
            progress = (time_sec - active_start) / actual_attack
            return event.weight * _ease_progress(progress, easing_mode)
        if actual_release > 0.0 and time_sec > event.end_sec:
            progress = (time_sec - event.end_sec) / actual_release
            return event.weight * (1.0 - _ease_progress(progress, easing_mode))
        return event.weight

    # Vowels use a bell window centred on the syllable: the mouth opens through
    # the first half, peaks at the vowel, and closes through the second half.
    # The window reaches 25% of the duration past each edge, so neighbouring
    # syllables overlap and crossfade instead of closing between phrases.
    lead = min(
        0.6,
        max(
            0.04,
            0.25 * duration,
            duration * (1.0 - _clamp(hold_ratio)) / 2.0,
            min(attack_sec, release_sec),
        ),
    )
    window_start = max(0.0, event.start_sec - lead)
    window_end = event.end_sec + lead
    if time_sec < window_start or time_sec > window_end:
        return 0.0
    center = 0.5 * (event.start_sec + event.end_sec)
    if time_sec <= center:
        progress = (time_sec - window_start) / max(1e-6, center - window_start)
    else:
        progress = (window_end - time_sec) / max(1e-6, window_end - center)
    return event.weight * _ease_progress(_clamp(progress), easing_mode)


def evaluate_viseme_channels(
    events: Iterable[VisemeEvent],
    time_sec: float,
    *,
    attack_ms: float = DEFAULT_ATTACK_MS,
    release_ms: float = DEFAULT_RELEASE_MS,
    hold_ratio: float = DEFAULT_HOLD_RATIO,
    easing_mode: str = "LINEAR",
    style: str = DEFAULT_MOTION_STYLE,
) -> Dict[str, float]:
    values = {channel: 0.0 for channel in VISEME_CHANNELS}
    attack_sec = max(0.0, attack_ms) / 1000.0
    release_sec = max(0.0, release_ms) / 1000.0
    for event in events:
        if event.viseme_id not in values:
            continue
        value = _clamp(
            _envelope(
                event,
                time_sec,
                attack_sec=attack_sec,
                release_sec=release_sec,
                hold_ratio=hold_ratio,
                easing_mode=easing_mode,
                style=style,
            )
        )
        values[event.viseme_id] = max(values[event.viseme_id], value)

    suppression = max(values["REST"], values["CLOSED"])
    for channel in VOWEL_CHANNELS:
        values[channel] *= 1.0 - suppression
    total = sum(values[channel] for channel in VOWEL_CHANNELS)
    if style == "NATURAL":
        # Compress only when the blend would exceed a full mouth; always
        # normalising cancels the envelopes and freezes the blend weights.
        if total > 1.0:
            for channel in VOWEL_CHANNELS:
                values[channel] /= total
    else:
        active = [channel for channel in VOWEL_CHANNELS if values[channel] > 0.0]
        if len(active) > 1 and total > 0.0:
            for channel in VOWEL_CHANNELS:
                values[channel] /= total
    return values


def _simplify_samples(samples: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(samples) <= 2:
        return list(samples)
    result = [samples[0]]
    for previous, current, following in zip(samples, samples[1:], samples[2:]):
        first_span = current[0] - previous[0]
        second_span = following[0] - current[0]
        first_slope = 0.0 if first_span == 0.0 else (current[1] - previous[1]) / first_span
        second_slope = (
            0.0 if second_span == 0.0 else (following[1] - current[1]) / second_span
        )
        if not isclose(first_slope, second_slope, abs_tol=1e-5):
            result.append(current)
    result.append(samples[-1])
    return result


def smooth_values(values: Sequence[float], sigma: float = 1.8) -> list[float]:
    """Gaussian smoothing over a per-frame value series (edges clamped).

    Speech mouth motion is continuous; sampling an envelope that has holds and
    short ramps produces visible single-frame steps.  A small blur turns those
    into the rounded transitions a mouth actually makes.
    """

    count = len(values)
    if sigma <= 0.0 or count < 3:
        return [float(value) for value in values]
    radius = max(1, int(round(sigma * 2.5)))
    weights = [exp(-0.5 * (offset / sigma) ** 2) for offset in range(-radius, radius + 1)]
    total = sum(weights)
    smoothed = []
    for index in range(count):
        accumulator = 0.0
        for offset, weight in zip(range(-radius, radius + 1), weights):
            neighbour = min(count - 1, max(0, index + offset))
            accumulator += float(values[neighbour]) * weight
        smoothed.append(accumulator / total)
    return smoothed


def sample_viseme_channels(
    events: Sequence[VisemeEvent],
    *,
    duration_sec: float,
    fps: float,
    attack_ms: float = DEFAULT_ATTACK_MS,
    release_ms: float = DEFAULT_RELEASE_MS,
    hold_ratio: float = DEFAULT_HOLD_RATIO,
    easing_mode: str = "LINEAR",
    style: str = DEFAULT_MOTION_STYLE,
    smooth_sigma: float | None = None,
) -> Mapping[str, list[tuple[float, float]]]:
    if fps <= 0.0:
        raise ValueError("effective FPS must be positive")
    if smooth_sigma is None:
        smooth_sigma = 1.8 if style == "NATURAL" else 0.0
    event_list = list(events)
    release_sec = max(0.0, release_ms) / 1000.0
    has_outer_release = easing_mode != "LINEAR" or any(
        event.viseme_id not in VOWEL_CHANNELS for event in event_list
    )
    transition_tail = release_sec if has_outer_release else 0.0
    timeline_end = max(
        max(0.0, duration_sec),
        max(
            (event.end_sec + transition_tail for event in event_list),
            default=0.0,
        ),
    )
    frame_count = max(1, int(ceil(timeline_end * fps)))
    sampled = {channel: [] for channel in VISEME_CHANNELS}
    for frame in range(frame_count + 1):
        values = evaluate_viseme_channels(
            event_list,
            frame / fps,
            attack_ms=attack_ms,
            release_ms=release_ms,
            hold_ratio=hold_ratio,
            easing_mode=easing_mode,
            style=style,
        )
        for channel in VISEME_CHANNELS:
            sampled[channel].append((float(frame), values[channel]))

    result = {}
    for channel, points in sampled.items():
        if smooth_sigma > 0.0 and len(points) >= 3:
            raw = [value for _frame, value in points]
            blurred = smooth_values(raw, smooth_sigma)
            # Keep true silence at zero: blurring must not leak motion into
            # frames where the timeline has no active event.
            points = [
                (frame, 0.0 if raw_value == 0.0 else max(0.0, min(1.0, value)))
                for (frame, _), raw_value, value in zip(points, raw, blurred)
            ]
        result[channel] = _simplify_samples(points)
    return result


__all__ = [
    "VISEME_CHANNELS",
    "VOWEL_CHANNELS",
    "build_viseme_events",
    "evaluate_viseme_channels",
    "sample_viseme_channels",
    "smooth_values",
]
