"""Lyric files, language annotation, and lyric-driven viseme generation.

This module is Blender-independent.  It turns user-supplied lyrics into the
same ``RecognitionDocument`` a speech recognizer would produce, so the whole
existing pipeline (G2P -> viseme events -> Blender bake) works unchanged.

Why lyrics instead of recognition: singing breaks speech recognizers (music,
pitch, melisma, long vowels, and hallucination on instrumental sections).
Providing the words and only solving *timing* is both more accurate and more
predictable.  The supported inputs are:

* LRC  - one timestamped line at a time, language tag optional
* QRC  - karaoke lyrics with per-character timestamps in milliseconds
* JSON - the aligner interchange format (``version: 1``), optional word level

Language handling: every word carries its own ``language_code``, so a single
song may freely mix ``zh-CN``, ``ja-JP``, and ``en-US``.  An explicit tag always
wins; otherwise the script is detected from the characters themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from ..constants import DEFAULT_ATTACK_MS, DEFAULT_RELEASE_MS
from ..core.schema import (
    LanguageSegment,
    PhonemeSegment,
    RecognitionDocument,
    WordSegment,
)
from ..core.timeline import build_viseme_events
from ..language.g2p import phonemize_words

SUPPORTED_LANGUAGE_CODES = ("zh-CN", "ja-JP", "en-US")
LYRICS_BACKEND_ID = "lyrics"

_LANGUAGE_ALIASES = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "cn": "zh-CN",
    "chs": "zh-CN",
    "chinese": "zh-CN",
    "ja": "ja-JP",
    "ja-jp": "ja-JP",
    "jp": "ja-JP",
    "japanese": "ja-JP",
    "en": "en-US",
    "en-us": "en-US",
    "eng": "en-US",
    "english": "en-US",
}

_TIME_TAG = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_META_TAG = re.compile(r"^\[(ti|ar|al|au|by|offset|re|ve|length)\s*:(.*)\]$", re.I)
_INLINE_LANGUAGE = re.compile(
    r"^\s*(?:[\[{<]\s*([A-Za-z]{2,8})\s*[\]}>]|([A-Za-z]{2,8})\s*[:：])\s*"
)
_QRC_LINE = re.compile(r"^\[(\d+),(\d+)\](.*)$")
_QRC_WORD = re.compile(r"([^\(\)\[\]]+?)\((\d+),(\d+)\)")

_HAN = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_KANA = "\u3041-\u3096\u30a1-\u30fa\u30fc\u31f0-\u31ff"
_HANGUL = "\uac00-\ud7af\u1100-\u11ff"
_LATIN = "A-Za-z\u00c0-\u024f"
_HAN_RE = re.compile(f"[{_HAN}]")
_KANA_RE = re.compile(f"[{_KANA}]")
_HANGUL_RE = re.compile(f"[{_HANGUL}]")
_LATIN_RE = re.compile(f"[{_LATIN}]")
_VOWEL_GROUP = re.compile(r"[aeiouy\u00e6\u00f8\u00e5\u0259\u026a\u028a]+", re.I)


class LyricsError(ValueError):
    """Raised when lyric input is malformed or unsupported."""


@dataclass
class LyricWord:
    """One timed unit of lyric text with an explicit output language."""

    text: str
    start_sec: float
    end_sec: float
    language_code: str = ""
    reading: str = ""

    def validate(self, context: str = "") -> None:
        label = f" ({context})" if context else ""
        if not self.text.strip():
            raise LyricsError(f"lyric word text must not be empty{label}")
        if self.start_sec < 0.0:
            raise LyricsError(f"lyric word start must be non-negative{label}")
        if self.end_sec < self.start_sec:
            raise LyricsError(f"lyric word end precedes start{label}")


@dataclass
class LyricLine:
    """One lyric line; ``words`` is filled by the parsers when known."""

    start_sec: float
    end_sec: float
    text: str = ""
    language_code: str = ""
    words: List[LyricWord] = field(default_factory=list)


@dataclass
class LyricsDocument:
    """Parsed lyrics in canonical seconds, before G2P."""

    lines: List[LyricLine] = field(default_factory=list)
    audio_duration_sec: float = 0.0
    source_format: str = "lrc"
    warnings: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.lines:
            raise LyricsError("lyric document contains no timed lines")
        previous_end = -1.0
        for index, line in enumerate(self.lines):
            if line.start_sec < 0.0:
                raise LyricsError(f"line {index} start must be non-negative")
            if line.end_sec < line.start_sec:
                raise LyricsError(f"line {index} end precedes start")
            if line.start_sec < previous_end - 1e-6:
                raise LyricsError(f"line {index} overlaps the previous line")
            previous_end = line.end_sec
            for word in line.words:
                word.validate(context=f"line {index}")


def normalise_language_code(value: str) -> str:
    """Map a user-facing alias to a supported language, or return ``""``."""

    key = str(value or "").strip().lower()
    if not key:
        return ""
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    for code in SUPPORTED_LANGUAGE_CODES:
        if key == code.lower():
            return code
    return ""


def detect_language(text: str) -> str:
    """Detect one language from the dominant script, or return ``""``."""

    if _KANA_RE.search(text):
        return "ja-JP"
    if _HANGUL_RE.search(text):
        raise LyricsError(
            "Korean lyrics are not supported; use one of "
            + ", ".join(SUPPORTED_LANGUAGE_CODES)
        )
    if _HAN_RE.search(text):
        return "zh-CN"
    if _LATIN_RE.search(text):
        return "en-US"
    return ""


def _char_class(character: str) -> str:
    if _HAN_RE.match(character):
        return "han"
    if _KANA_RE.match(character):
        return "kana"
    if _HANGUL_RE.match(character):
        return "hangul"
    if _LATIN_RE.match(character):
        return "latin"
    return "other"


def _english_weight(word: str) -> float:
    groups = _VOWEL_GROUP.findall(word)
    return float(max(1, len(groups)))


def tokenize_line(text: str, *, default_language_code: str = "") -> List[tuple]:
    """Split one lyric line into ``(text, language_code)`` units.

    Chinese is split per character because a character is the syllable the
    mouth follows.  Japanese is kept in chunks so OpenJTalk can resolve kanji
    readings from context; splitting per character would read ``今日`` as
    ``いまひ``.  Latin runs become one word each.
    """

    text = text.strip()
    if not text:
        raise LyricsError("cannot tokenize an empty lyric line")

    has_kana = bool(_KANA_RE.search(text)) or normalise_language_code(
        default_language_code
    ) == "ja-JP"
    tokens: List[tuple] = []
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        kind = _char_class(character)
        if kind == "hangul":
            raise LyricsError(
                "Korean lyrics are not supported; use one of "
                + ", ".join(SUPPORTED_LANGUAGE_CODES)
            )
        if kind == "other":
            index += 1
            continue
        if kind == "latin":
            end = index
            while end < length and (
                _char_class(text[end]) == "latin" or text[end] in "'\u2019"
            ):
                end += 1
            tokens.append((text[index:end], "en-US"))
            index = end
            continue
        if kind == "kana":
            has_kana = True
        if has_kana:
            end = index
            while end < length and _char_class(text[end]) in ("han", "kana"):
                end += 1
            tokens.append((text[index:end], "ja-JP"))
            index = end
            continue
        language = "zh-CN"
        tokens.append((character, language))
        index += 1

    if not tokens:
        raise LyricsError(f"lyric line has no pronounceable text: {text!r}")
    return tokens


def _token_weight(text: str, language_code: str) -> float:
    if language_code == "en-US":
        return _english_weight(text)
    if language_code == "ja-JP":
        return float(max(1, len(text)))
    return 1.0


def allocate_spans(
    start_sec: float,
    end_sec: float,
    weights: Sequence[float],
) -> List[tuple]:
    """Split ``[start_sec, end_sec]`` into contiguous spans by weight."""

    if not weights:
        return []
    span = max(0.0, end_sec - start_sec)
    total = sum(weights)
    if total <= 0.0:
        return [(start_sec, start_sec) for _ in weights]
    spans: List[tuple] = []
    cursor = start_sec
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            spans.append((cursor, end_sec))
            break
        next_cursor = cursor + span * weight / total
        spans.append((cursor, max(cursor, next_cursor)))
        cursor = next_cursor
    return spans


def _estimate_line_duration(text: str) -> float:
    return max(0.6, 0.22 * float(len(text.strip())))


def _split_inline_language(body: str, default_language_code: str) -> tuple:
    """Strip a leading ``[zh]`` / ``{ja}`` / ``en:`` tag from a line body."""

    default = normalise_language_code(default_language_code)
    match = _INLINE_LANGUAGE.match(body)
    if match is None:
        return default, body.strip()
    name = match.group(1) or match.group(2) or ""
    language = normalise_language_code(name)
    if not language:
        return default, body.strip()
    remainder = body[match.end() :].strip()
    if not remainder:
        return default, body.strip()
    return language, remainder


def _fraction_to_seconds(value: str) -> float:
    if not value:
        return 0.0
    digits = value[:3].ljust(3, "0")
    return float("0." + digits)


def _fill_line_words(
    line: LyricLine,
    *,
    default_language_code: str,
) -> None:
    if line.words:
        for word in line.words:
            if not word.language_code:
                word.language_code = (
                    normalise_language_code(line.language_code)
                    or normalise_language_code(default_language_code)
                    or detect_language(word.text)
                )
        line.words.sort(key=lambda value: value.start_sec)
        return

    text = line.text.strip()
    if not text:
        line.words = []
        return
    line_language = normalise_language_code(line.language_code) or normalise_language_code(
        default_language_code
    )
    try:
        tokens = tokenize_line(text, default_language_code=line_language)
    except LyricsError:
        raise
    weights = [_token_weight(token, language) for token, language in tokens]
    spans = allocate_spans(line.start_sec, line.end_sec, weights)
    line.words = [
        LyricWord(
            text=token,
            start_sec=span[0],
            end_sec=span[1],
            language_code=language,
        )
        for (token, language), span in zip(tokens, spans)
    ]


def _entries_to_lines(
    entries: Sequence[tuple],
    *,
    default_language_code: str,
    audio_duration_sec: float,
    gap_sec: float,
) -> List[LyricLine]:
    lines: List[LyricLine] = []
    for index, (start_sec, body, language) in enumerate(entries):
        if index + 1 < len(entries):
            end_sec = max(start_sec, entries[index + 1][0] - max(0.0, gap_sec))
        else:
            end_sec = (
                max(start_sec, audio_duration_sec)
                if audio_duration_sec > start_sec
                else start_sec + _estimate_line_duration(body)
            )
        lines.append(
            LyricLine(
                start_sec=start_sec,
                end_sec=end_sec,
                text=body,
                language_code=language or normalise_language_code(default_language_code),
            )
        )
    return lines


def parse_lrc(
    text: str,
    *,
    default_language_code: str = "",
    audio_duration_sec: float = 0.0,
    gap_sec: float = 0.0,
) -> LyricsDocument:
    """Parse LRC text into lines with optional language tags.

    The ``[offset:ms]`` tag follows the common convention that a positive
    value moves the lyrics earlier by that many milliseconds.
    """

    offset_sec = 0.0
    warnings: List[str] = []
    entries: List[tuple] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        meta = _META_TAG.match(stripped)
        if meta:
            name = meta.group(1).lower()
            value = meta.group(2).strip()
            if name == "offset":
                try:
                    offset_sec = -float(value) / 1000.0
                except ValueError:
                    warnings.append(f"invalid LRC offset value: {value!r}")
            continue
        stamps = _TIME_TAG.findall(stripped)
        if not stamps:
            continue
        body = _TIME_TAG.sub("", stripped).strip()
        language, body = _split_inline_language(body, default_language_code)
        if not body:
            continue
        for minutes, seconds, fraction in stamps:
            start_sec = (
                int(minutes) * 60.0 + int(seconds) + _fraction_to_seconds(fraction)
            )
            start_sec = max(0.0, start_sec + offset_sec)
            entries.append((start_sec, body, language))

    entries.sort(key=lambda item: item[0])
    lines = _entries_to_lines(
        entries,
        default_language_code=default_language_code,
        audio_duration_sec=audio_duration_sec,
        gap_sec=gap_sec,
    )
    document = LyricsDocument(
        lines=lines,
        audio_duration_sec=max(0.0, audio_duration_sec),
        source_format="lrc",
        warnings=warnings,
    )
    document.validate()
    return document


def parse_qrc(text: str, *, default_language_code: str = "") -> LyricsDocument:
    """Parse QRC karaoke lyrics (per-character timestamps in milliseconds)."""

    lines: List[LyricLine] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        match = _QRC_LINE.match(stripped)
        if match is None:
            continue
        body = match.group(3)
        words: List[LyricWord] = []
        for word_match in _QRC_WORD.finditer(body):
            word_text = word_match.group(1).strip()
            if not word_text:
                continue
            start_sec = int(word_match.group(2)) / 1000.0
            duration = int(word_match.group(3)) / 1000.0
            if duration <= 0.0:
                duration = 0.18
            words.append(
                LyricWord(
                    text=word_text,
                    start_sec=start_sec,
                    end_sec=start_sec + duration,
                    language_code=normalise_language_code(default_language_code)
                    or detect_language(word_text),
                )
            )
        if not words:
            continue
        lines.append(
            LyricLine(
                start_sec=min(word.start_sec for word in words),
                end_sec=max(word.end_sec for word in words),
                text="".join(word.text for word in words),
                words=words,
            )
        )

    lines.sort(key=lambda value: value.start_sec)
    document = LyricsDocument(lines=lines, source_format="qrc")
    document.validate()
    return document


def _line_from_payload(
    raw: Mapping[str, Any],
    *,
    default_language_code: str,
    index: int,
) -> LyricLine:
    if not isinstance(raw, Mapping):
        raise LyricsError(f"line {index} must be an object")
    try:
        start_sec = float(raw["start"])
        end_sec = float(raw["end"])
    except KeyError as exc:
        raise LyricsError(f"line {index} requires start and end") from exc
    if end_sec < start_sec:
        raise LyricsError(f"line {index} end precedes start")

    raw_words = raw.get("words")
    words: List[LyricWord] = []
    if raw_words is not None:
        if not isinstance(raw_words, list):
            raise LyricsError(f"line {index} words must be a list")
        for word_index, raw_word in enumerate(raw_words):
            if not isinstance(raw_word, Mapping):
                raise LyricsError(
                    f"line {index} word {word_index} must be an object"
                )
            try:
                word_start = float(raw_word["start"])
                word_end = float(raw_word["end"])
            except KeyError as exc:
                raise LyricsError(
                    f"line {index} word {word_index} requires start and end"
                ) from exc
            words.append(
                LyricWord(
                    text=str(raw_word.get("text", "")),
                    start_sec=word_start,
                    end_sec=word_end,
                    language_code=normalise_language_code(
                        str(raw_word.get("lang", ""))
                    ),
                    reading=str(raw_word.get("reading", "")),
                )
            )

    language = normalise_language_code(str(raw.get("lang", "")))
    text = str(raw.get("text", "")).strip()
    if not text and words:
        text = "".join(word.text for word in words)
    return LyricLine(
        start_sec=start_sec,
        end_sec=end_sec,
        text=text,
        language_code=language
        or normalise_language_code(default_language_code),
        words=words,
    )


def parse_align_json(
    data: Mapping[str, Any],
    *,
    default_language_code: str = "",
) -> LyricsDocument:
    """Parse the aligner interchange format (``version: 1``)."""

    if not isinstance(data, Mapping):
        raise LyricsError("align JSON root must be an object")

    version = data.get("version", 1)
    if int(version) != 1:
        raise LyricsError(f"unsupported align JSON version: {version!r}")

    document_language = normalise_language_code(
        str(data.get("language", ""))
    ) or normalise_language_code(default_language_code)
    audio_duration_sec = float(data.get("audio_duration_sec", 0.0) or 0.0)

    raw_flat_words = data.get("words")
    if raw_flat_words is not None:
        words: List[LyricWord] = []
        for index, raw_word in enumerate(raw_flat_words):
            if not isinstance(raw_word, Mapping):
                raise LyricsError(f"word {index} must be an object")
            try:
                start_sec = float(raw_word["start"])
                end_sec = float(raw_word["end"])
            except KeyError as exc:
                raise LyricsError(f"word {index} requires start and end") from exc
            words.append(
                LyricWord(
                    text=str(raw_word.get("text", "")),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    language_code=normalise_language_code(
                        str(raw_word.get("lang", ""))
                    )
                    or document_language,
                    reading=str(raw_word.get("reading", "")),
                )
            )
        words.sort(key=lambda value: value.start_sec)
        document = LyricsDocument(
            lines=[
                LyricLine(
                    start_sec=min(word.start_sec for word in words),
                    end_sec=max(word.end_sec for word in words),
                    text="".join(word.text for word in words),
                    language_code=document_language,
                    words=words,
                )
            ]
            if words
            else [],
            audio_duration_sec=audio_duration_sec,
            source_format="json",
        )
        document.validate()
        return document

    raw_lines = data.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise LyricsError("align JSON requires a non-empty lines or words list")
    lines = [
        _line_from_payload(
            raw,
            default_language_code=document_language,
            index=index,
        )
        for index, raw in enumerate(raw_lines)
    ]
    lines.sort(key=lambda value: value.start_sec)
    document = LyricsDocument(
        lines=lines,
        audio_duration_sec=audio_duration_sec,
        source_format="json",
    )
    document.validate()
    return document


def distribute_lines(
    raw_lines: Sequence[str],
    start_sec: float,
    end_sec: float,
) -> List[LyricLine]:
    """Lay untimed plain-text lines over a span, weighted by length.

    This is the explicit fallback for lyrics without timestamps: it keeps the
    pipeline usable and makes the missing alignment obvious instead of
    pretending a uniform layout is a real alignment.
    """

    cleaned = [line.strip() for line in raw_lines if line.strip()]
    if not cleaned:
        raise LyricsError("plain-text lyrics contain no usable lines")
    weights = [max(1, len(line)) for line in cleaned]
    total = float(sum(weights))
    span = max(0.1, end_sec - start_sec)
    lines: List[LyricLine] = []
    cursor = start_sec
    for index, (line, weight) in enumerate(zip(cleaned, weights)):
        if index == len(cleaned) - 1:
            line_end = end_sec
        else:
            line_end = cursor + span * weight / total
        lines.append(
            LyricLine(
                start_sec=cursor,
                end_sec=max(cursor + 0.05, line_end),
                text=line,
            )
        )
        cursor = line_end
    return lines


def parse_plain_lines(
    text: str,
    *,
    start_sec: float,
    end_sec: float,
    default_language_code: str = "",
) -> LyricsDocument:
    """Build a document from untimed plain-text lyrics."""

    lines = distribute_lines(text.splitlines(), start_sec, end_sec)
    for line in lines:
        line.language_code = normalise_language_code(default_language_code)
    document = LyricsDocument(
        lines=lines,
        audio_duration_sec=max(0.0, end_sec),
        source_format="plain",
    )
    document.validate()
    return document


def load_lyrics(
    path: str,
    *,
    default_language_code: str = "",
    audio_duration_sec: float = 0.0,
    gap_sec: float = 0.0,
) -> LyricsDocument:
    """Load an LRC, QRC, or align JSON lyric file by extension."""

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise LyricsError(f"lyric file does not exist: {file_path}")
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LyricsError(f"cannot read lyric file: {file_path}") from exc
    except UnicodeDecodeError as exc:
        raise LyricsError(
            f"lyric file is not UTF-8: {file_path}"
        ) from exc

    suffix = file_path.suffix.lower()
    if suffix == ".qrc":
        return parse_qrc(text, default_language_code=default_language_code)
    if suffix in {".json", ".align"}:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LyricsError(f"invalid lyric JSON: {file_path}") from exc
        return parse_align_json(
            payload, default_language_code=default_language_code
        )
    return parse_lrc(
        text,
        default_language_code=default_language_code,
        audio_duration_sec=audio_duration_sec,
        gap_sec=gap_sec,
    )


def to_word_segments(
    document: LyricsDocument,
    *,
    default_language_code: str = "",
) -> List[WordSegment]:
    """Flatten a lyric document into validated, language-tagged word records."""

    document.validate()
    default_language = normalise_language_code(default_language_code)
    segments: List[WordSegment] = []
    for line_index, line in enumerate(document.lines):
        if not line.words:
            _fill_line_words(line, default_language_code=default_language)
        for word in line.words:
            if word.start_sec < 0.0 or word.end_sec < word.start_sec:
                raise LyricsError(f"line {line_index} has an invalid word timing")
            language = (
                word.language_code
                or normalise_language_code(line.language_code)
                or detect_language(word.text)
                or default_language
            )
            segments.append(
                WordSegment(
                    text=word.text,
                    start_sec=word.start_sec,
                    end_sec=word.end_sec,
                    confidence=1.0,
                    raw_confidence=1.0,
                    language_code=language or default_language,
                    model_id=LYRICS_BACKEND_ID,
                )
            )
    if not segments:
        raise LyricsError("lyrics produced no timed words")
    segments.sort(key=lambda value: (value.start_sec, value.end_sec))
    for segment in segments:
        segment.validate()
    return segments


def _language_segments(words: Sequence[WordSegment]) -> List[LanguageSegment]:
    segments: List[LanguageSegment] = []
    for word in words:
        language = word.language_code or "und"
        if segments and segments[-1].language_code == language:
            segments[-1].end_sec = max(segments[-1].end_sec, word.end_sec)
            continue
        segments.append(
            LanguageSegment(
                start_sec=word.start_sec,
                end_sec=word.end_sec,
                language_code=language,
                model_id=LYRICS_BACKEND_ID,
                confidence=1.0,
                source="MODEL_SCORE",
            )
        )
    return segments


def run_lyrics_pipeline(
    words: Sequence[WordSegment],
    *,
    default_language_code: str = "",
    attack_ms: float = DEFAULT_ATTACK_MS,
    release_ms: float = DEFAULT_RELEASE_MS,
    audio_duration_sec: float = 0.0,
    phonemizer: Callable[..., List[PhonemeSegment]] | None = None,
    blend: bool = False,
) -> RecognitionDocument:
    """G2P the supplied words and build viseme events without recognition.

    ``blend`` builds simultaneous A/I/U/E/O weights per Chinese syllable, so a
    final such as ``ai`` becomes one blended mouth shape instead of a fast
    sequence of two single-vowel shapes.
    """

    prepared = [word for word in words if word.text.strip()]
    if not prepared:
        raise LyricsError("no lyric words to generate from")
    for word in prepared:
        if word.start_sec < 0.0:
            word.start_sec = 0.0
        if word.end_sec < word.start_sec:
            word.end_sec = word.start_sec
        word.validate()
    prepared.sort(key=lambda value: (value.start_sec, value.end_sec))

    known_languages = sorted(
        {word.language_code for word in prepared if word.language_code}
    )
    default_language = (
        normalise_language_code(default_language_code)
        or (known_languages[0] if known_languages else "zh-CN")
    )
    for word in prepared:
        if not word.language_code:
            word.language_code = default_language

    phonemize = phonemizer if phonemizer is not None else phonemize_words
    if blend:
        from ..language.blend import build_blended_timeline

        events, phonemes = build_blended_timeline(
            prepared,
            default_language_code=default_language,
            phonemizer=phonemize,
        )
    else:
        phonemes = phonemize(prepared, default_language_code=default_language)
        events = build_viseme_events(
            phonemes,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
    languages = sorted({word.language_code for word in prepared})
    duration = max(audio_duration_sec, prepared[-1].end_sec)

    document = RecognitionDocument(
        backend_id=LYRICS_BACKEND_ID,
        model_id=LYRICS_BACKEND_ID,
        language_code=languages[0] if len(languages) == 1 else "mixed",
        audio_duration_sec=duration,
        words=prepared,
        phonemes=phonemes,
        events=events,
        language_segments=_language_segments(prepared),
        metadata={
            "source": LYRICS_BACKEND_ID,
            "word_count": len(prepared),
            "languages": languages,
        },
    )
    document.validate()
    return document


def resolve_words(payload: Mapping[str, Any]) -> List[WordSegment]:
    """Resolve the words of a ``mode: "lyrics"`` worker job payload."""

    if not isinstance(payload, Mapping):
        raise LyricsError("job payload must be an object")
    default_language = str(payload.get("default_language_code", ""))
    audio_duration_sec = float(payload.get("audio_duration_sec", 0.0) or 0.0)

    raw_words = payload.get("words")
    if raw_words is not None:
        if not isinstance(raw_words, list) or not raw_words:
            raise LyricsError("job.words must be a non-empty list")
        document = parse_align_json(
            {
                "version": 1,
                "words": raw_words,
                "language": default_language,
                "audio_duration_sec": audio_duration_sec,
            },
            default_language_code=default_language,
        )
        return to_word_segments(
            document, default_language_code=default_language
        )

    raw_lines = payload.get("lines")
    if raw_lines is not None:
        if not isinstance(raw_lines, list) or not raw_lines:
            raise LyricsError("job.lines must be a non-empty list")
        lines = [
            _line_from_payload(
                raw,
                default_language_code=default_language,
                index=index,
            )
            for index, raw in enumerate(raw_lines)
        ]
        lines.sort(key=lambda value: value.start_sec)
        document = LyricsDocument(
            lines=lines,
            audio_duration_sec=audio_duration_sec,
            source_format="json",
        )
        return to_word_segments(
            document, default_language_code=default_language
        )

    lyrics_path = str(payload.get("lyrics_path", "")).strip()
    if lyrics_path:
        document = load_lyrics(
            lyrics_path,
            default_language_code=default_language,
            audio_duration_sec=audio_duration_sec,
            gap_sec=float(payload.get("gap_sec", 0.0) or 0.0),
        )
        return to_word_segments(
            document, default_language_code=default_language
        )

    raise LyricsError(
        "lyrics job requires one of lyrics_path, lines, or words"
    )


__all__ = [
    "LYRICS_BACKEND_ID",
    "SUPPORTED_LANGUAGE_CODES",
    "LyricLine",
    "LyricWord",
    "LyricsDocument",
    "LyricsError",
    "allocate_spans",
    "detect_language",
    "distribute_lines",
    "load_lyrics",
    "normalise_language_code",
    "parse_align_json",
    "parse_lrc",
    "parse_plain_lines",
    "parse_qrc",
    "resolve_words",
    "run_lyrics_pipeline",
    "to_word_segments",
    "tokenize_line",
]
