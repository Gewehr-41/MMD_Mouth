"""Romanization for CTC alignment: one Latin letter stream per language.

The MMS-FA acoustic model consumes romanised letters, so every language needs
its own mapping:

* Chinese  - pypinyin, then strip tone digits
* Japanese - OpenJTalk phone labels are already romaji ("k", "sh", "N"…)
* English  - the spelling itself (what uroman produces for Latin script)

Mixing them lets a single alignment pass cover 中英 / 日英 mixed lyrics instead
of reading every line with Chinese pinyin.
"""

from __future__ import annotations

import re

_LETTER = re.compile(r"[a-z]")
# Articulation-only labels carry no letters worth aligning.  Compared
# case-sensitively: uppercase "N" is the moraic nasal and *is* pronounced.
_SKIP_JAPANESE = {"cl", "pau", "sil", "sp"}


def _latin_letters(text: str) -> list[str]:
    return _LETTER.findall(text.lower())


def _zh_letters(text: str) -> list[str]:
    from pypinyin import Style, lazy_pinyin

    letters: list[str] = []
    for syllable in lazy_pinyin(text, style=Style.NORMAL, errors="ignore"):
        letters.extend(_latin_letters(syllable.replace("ü", "v")))
    return letters


def _ja_letters(text: str) -> list[str]:
    from .g2p import pronunciation_tokens

    letters: list[str] = []
    for token in pronunciation_tokens(text, "ja-JP"):
        source = str(token.source)
        if source in _SKIP_JAPANESE:
            continue
        letters.extend(_latin_letters(source))
    return letters


def romanize_letters(text: str, language_code: str = "") -> list[str]:
    """Return the Latin letters used to align ``text``."""

    family = str(language_code or "").split("-")[0].lower()
    if family == "zh":
        return _zh_letters(text)
    if family == "ja":
        letters = _ja_letters(text)
        if letters:
            return letters
        # Simplified characters (e.g. 爱) may be unknown to OpenJTalk; fall back
        # to pinyin so the character still gets an alignment anchor.
        return _zh_letters(text)
    return _latin_letters(text)


__all__ = ["romanize_letters"]
