"""Pinyin syllable -> simultaneous A/I/U/E/O blend weights.

MMD models only carry the five Japanese vowel morphs, so a Chinese final such as
``ai`` cannot be a single morph.  Instead every field is expressed as weights on
the five channels, and the VMD keys all of them at once: ``ai`` becomes
``A 0.6 + I 0.4``, which MMD blends additively.

Weights on one syllable sum to about 1.0, matching the timeline evaluator's
normalisation, so the mouth opens by the same amount as a single vowel.
"""

from __future__ import annotations

INITIALS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)

# final -> {channel: weight}
BLENDS: dict[str, dict[str, float]] = {
    # single vowels
    "a": {"A": 1.0},
    "o": {"O": 1.0},
    "e": {"E": 1.0},
    "i": {"I": 1.0},
    "u": {"U": 1.0},
    "v": {"U": 0.7, "I": 0.3},
    "er": {"A": 0.45, "E": 0.35, "O": 0.2},
    # diphthongs
    "ai": {"A": 0.6, "I": 0.4},
    "ei": {"E": 0.6, "I": 0.4},
    "ao": {"A": 0.6, "O": 0.4},
    "ou": {"O": 0.6, "U": 0.4},
    # i-medial
    "ia": {"I": 0.3, "A": 0.7},
    "ie": {"I": 0.3, "E": 0.7},
    "iao": {"I": 0.25, "A": 0.5, "O": 0.25},
    "iu": {"I": 0.3, "O": 0.35, "U": 0.35},
    "ian": {"I": 0.25, "A": 0.75},
    "iang": {"I": 0.2, "A": 0.8},
    "in": {"I": 0.9},
    "ing": {"I": 0.9},
    "iong": {"I": 0.2, "O": 0.8},
    # u-medial
    "ua": {"U": 0.3, "A": 0.7},
    "uo": {"U": 0.4, "O": 0.6},
    "uai": {"U": 0.25, "A": 0.5, "I": 0.25},
    "ui": {"U": 0.35, "E": 0.3, "I": 0.35},
    "uan": {"U": 0.25, "A": 0.75},
    "uang": {"U": 0.2, "A": 0.8},
    "un": {"U": 0.88},
    "ueng": {"U": 0.3, "E": 0.7},
    # v-medial
    "ve": {"U": 0.4, "E": 0.6},
    "van": {"U": 0.4, "E": 0.45, "A": 0.15},
    "vn": {"U": 0.7, "I": 0.3},
    # nasal finals
    "an": {"A": 0.85},
    "en": {"E": 0.85},
    "ang": {"A": 0.9},
    "eng": {"E": 0.9},
    "ong": {"O": 0.9},
}

# zh/ch/sh/r/z/c/s + i is the apical vowel, closer to I than to a pure i.
APICAL_INITIALS = {"zh", "ch", "sh", "r", "z", "c", "s"}
BILABIAL_INITIALS = {"b", "p", "m"}
# y/w are spelling conventions for glides, not consonants that close the mouth,
# so they must not steal time from the vowel.
GLIDE_INITIALS = {"y", "w"}

DEFAULT_BLEND = {"A": 1.0}


def split_syllable(syllable: str) -> tuple[str, str]:
    """Split a toneless pinyin syllable into ``(initial, final)``.

    Orthographic ``y``/``w`` spellings are rewritten to their real medials so
    ``wo`` behaves like ``uo`` and ``yan`` like ``ian``.
    """

    text = str(syllable or "").strip().lower()
    text = text.replace("ü", "v").replace("u:", "v")
    for initial in INITIALS:
        if text.startswith(initial):
            final = text[len(initial) :]
            if initial in {"j", "q", "x", "y"} and final.startswith("u"):
                final = "v" + final[1:]
            return initial, _normalise_final(initial, final)
    return "", text


_W_MAP = {
    "o": "uo",
    "a": "ua",
    "ai": "uai",
    "ei": "ui",
    "an": "uan",
    "en": "un",
    "ang": "uang",
    "eng": "ueng",
}

_Y_MAP = {
    "a": "ia",
    "e": "ie",
    "ao": "iao",
    "ou": "iu",
    "an": "ian",
    "in": "in",
    "ang": "iang",
    "ing": "ing",
    "ong": "iong",
}


def _normalise_final(initial: str, final: str) -> str:
    if initial == "w":
        return _W_MAP.get(final, final)
    if initial == "y":
        return _Y_MAP.get(final, final)
    return final


def blend_for_syllable(syllable: str) -> tuple[str, str, dict[str, float]]:
    """Return ``(initial, final, weights)`` for one syllable."""

    initial, final = split_syllable(syllable)
    if not final:
        final = "e"
    if initial in APICAL_INITIALS and final == "i":
        return initial, final, {"I": 0.8, "E": 0.2}
    weights = BLENDS.get(final)
    if weights is None:
        # Unknown final (foreign syllable): fall back to its first vowel.
        for letter in final:
            weights = BLENDS.get(letter)
            if weights:
                break
    return initial, final, dict(weights or DEFAULT_BLEND)


def pinyin_of(text: str) -> str:
    """Toneless pinyin of a word, or ``""`` when it is not Chinese."""

    from pypinyin import Style, lazy_pinyin

    parts = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
    return parts[0] if parts else ""


def _zh_events(word, index: int) -> list:
    """Build simultaneous channel events for one Chinese character."""

    from ..core.schema import VisemeEvent

    text = str(word.text).strip()
    if not text:
        return []
    syllable = pinyin_of(text)
    initial, _final, weights = blend_for_syllable(syllable)
    start = max(0.0, float(word.start_sec))
    end = max(start, float(word.end_sec))
    if end - start <= 0.0:
        return []

    duration = end - start
    consonant = (
        min(0.12, duration * 0.35)
        if initial and initial not in GLIDE_INITIALS
        else 0.0
    )
    common = {
        "source": "G2P",
        "source_index": index,
        "source_text": text,
        "phoneme": syllable,
        "language_code": "zh-CN",
    }
    events = []
    if initial in BILABIAL_INITIALS:
        events.append(
            VisemeEvent(
                viseme_id="CLOSED",
                start_sec=start,
                end_sec=start + consonant,
                weight=1.0,
                priority=100,
                **common,
            )
        )
    elif initial == "f":
        events.append(
            VisemeEvent(
                viseme_id="CLOSED",
                start_sec=start,
                end_sec=start + consonant,
                weight=0.45,
                priority=90,
                **common,
            )
        )
    for channel, weight in weights.items():
        if weight <= 0.0:
            continue
        events.append(
            VisemeEvent(
                viseme_id=channel,
                start_sec=start + consonant,
                end_sec=end,
                weight=float(weight),
                priority=50,
                **common,
            )
        )
    return events


def build_blended_timeline(words, *, default_language_code: str, phonemizer=None):
    """Return ``(events, phonemes)`` for a blended timeline.

    Chinese words become blended events; other languages keep their phoneme
    G2P, so a mixed clip still reports phonemes for the non-Chinese lines.
    """

    from ..core.timeline import build_viseme_events

    if phonemizer is None:
        from .g2p import phonemize_words as phonemizer

    events = []
    phonemes = []
    for index, word in enumerate(words):
        language = (word.language_code or default_language_code or "").lower()
        if language.startswith("zh"):
            events.extend(_zh_events(word, index))
            continue
        word_phonemes = phonemizer(
            [word], default_language_code=language or default_language_code
        )
        phonemes.extend(word_phonemes)
        events.extend(build_viseme_events(word_phonemes))
    events.sort(key=lambda event: (event.start_sec, -event.priority))
    return events, phonemes


def build_blended_events(words, *, default_language_code: str, phonemizer=None) -> list:
    """Build the viseme timeline with simultaneous channel blends for Chinese."""

    events, _phonemes = build_blended_timeline(
        words,
        default_language_code=default_language_code,
        phonemizer=phonemizer,
    )
    return events


__all__ = [
    "APICAL_INITIALS",
    "BILABIAL_INITIALS",
    "BLENDS",
    "DEFAULT_BLEND",
    "blend_for_syllable",
    "build_blended_events",
    "pinyin_of",
    "split_syllable",
]
