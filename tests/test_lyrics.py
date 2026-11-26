import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mmd_mouth.core.schema import PhonemeSegment, WordSegment
from mmd_mouth.recognition import lyrics

try:
    import pypinyin  # noqa: F401

    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


def stub_phonemize(words, *, default_language_code):
    """Deterministic stand-in for the real G2P layer."""

    segments = []
    for word in words:
        closed = word.text.lower().startswith(("m", "b", "p"))
        segments.append(
            PhonemeSegment(
                phoneme="m" if closed else "a",
                source_phoneme="m" if closed else "a",
                start_sec=word.start_sec,
                end_sec=word.end_sec,
                source_text=word.text,
                phoneme_type="CONSONANT" if closed else "VOWEL",
                articulation_class="BILABIAL_CLOSURE" if closed else "VOWEL",
                viseme_id="CLOSED" if closed else "A",
                close_strength=1.0 if closed else 0.0,
                confidence=1.0,
                language_code=word.language_code,
            )
        )
    return segments


class LrcParsingTests(unittest.TestCase):
    def test_metadata_is_ignored_and_offset_is_applied(self):
        document = lyrics.parse_lrc(
            "[ti:Lunar]\n[ar:Nobody]\n[offset:+500]\n[00:01.20]月亮代表我的心\n",
            default_language_code="zh-CN",
        )

        self.assertEqual(len(document.lines), 1)
        self.assertAlmostEqual(document.lines[0].start_sec, 0.7)

    def test_one_line_with_two_timestamps_becomes_two_lines(self):
        document = lyrics.parse_lrc(
            "[00:05.00][00:20.00]La la\n",
            default_language_code="en-US",
            audio_duration_sec=30.0,
        )

        self.assertEqual(
            [round(line.start_sec, 2) for line in document.lines], [5.0, 20.0]
        )
        self.assertAlmostEqual(document.lines[0].end_sec, 20.0)
        self.assertAlmostEqual(document.lines[1].end_sec, 30.0)

    def test_inline_language_tags_override_the_default(self):
        document = lyrics.parse_lrc(
            "[00:02.00][ja]愛してる\n[00:04.00]{en}hello\n[00:06.00]月亮\n",
            default_language_code="zh-CN",
        )

        self.assertEqual(
            [line.language_code for line in document.lines],
            ["ja-JP", "en-US", "zh-CN"],
        )
        self.assertEqual(document.lines[0].text, "愛してる")

    def test_last_line_gets_an_estimated_duration_without_audio_length(self):
        document = lyrics.parse_lrc("[00:10.00]only line\n")

        line = document.lines[0]
        self.assertGreater(line.end_sec, line.start_sec)

    def test_invalid_offset_is_reported_as_a_warning(self):
        document = lyrics.parse_lrc(
            "[offset:abc]\n[00:01.00]text\n", default_language_code="en-US"
        )

        self.assertEqual(len(document.lines), 1)
        self.assertTrue(document.warnings)


class TokenizerTests(unittest.TestCase):
    def test_chinese_splits_per_character(self):
        tokens = lyrics.tokenize_line("月亮代表", default_language_code="zh-CN")

        self.assertEqual([text for text, _ in tokens], ["月", "亮", "代", "表"])
        self.assertTrue(all(lang == "zh-CN" for _, lang in tokens))

    def test_japanese_keeps_context_chunks(self):
        tokens = lyrics.tokenize_line("君の名前を呼ぶ")

        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0][1], "ja-JP")

    def test_japanese_language_keeps_kanji_only_lines_together(self):
        tokens = lyrics.tokenize_line("東京", default_language_code="ja-JP")

        self.assertEqual(tokens, [("東京", "ja-JP")])

    def test_english_splits_on_word_boundaries(self):
        tokens = lyrics.tokenize_line("I really love you", default_language_code="en-US")

        self.assertEqual(len(tokens), 4)
        self.assertTrue(all(lang == "en-US" for _, lang in tokens))

    def test_mixed_line_is_split_by_script(self):
        tokens = lyrics.tokenize_line("我爱你 Baby", default_language_code="zh-CN")

        self.assertEqual([text for text, _ in tokens], ["我", "爱", "你", "Baby"])
        self.assertEqual(tokens[-1][1], "en-US")

    def test_punctuation_is_dropped(self):
        tokens = lyrics.tokenize_line("你好，世界！", default_language_code="zh-CN")

        self.assertEqual([text for text, _ in tokens], ["你", "好", "世", "界"])

    def test_korean_is_rejected(self):
        with self.assertRaises(lyrics.LyricsError):
            lyrics.tokenize_line("안녕하세요")


class SpanAllocationTests(unittest.TestCase):
    def test_spans_cover_the_line_in_weight_order(self):
        spans = lyrics.allocate_spans(1.0, 3.0, [1.0, 1.0, 2.0])

        self.assertAlmostEqual(spans[0][0], 1.0)
        self.assertAlmostEqual(spans[0][1], 1.5)
        self.assertAlmostEqual(spans[1][1], 2.0)
        self.assertAlmostEqual(spans[2][1], 3.0)

    def test_zero_length_line_produces_empty_spans(self):
        spans = lyrics.allocate_spans(2.0, 2.0, [1.0, 1.0])

        self.assertEqual(spans, [(2.0, 2.0), (2.0, 2.0)])

    def test_spans_never_overlap(self):
        spans = lyrics.allocate_spans(0.0, 1.0, [3.0, 1.0, 2.0])

        for previous, current in zip(spans, spans[1:]):
            self.assertLessEqual(previous[1], current[0] + 1e-9)


class QrcParsingTests(unittest.TestCase):
    def test_per_character_timestamps_are_converted_to_seconds(self):
        document = lyrics.parse_qrc(
            "[1000,2000]月(1000,400)亮(1400,400)代(1800,500)\n"
            "[3500,1500]我(3500,500)爱(4000,500)\n",
            default_language_code="zh-CN",
        )

        self.assertEqual(len(document.lines), 2)
        first = document.lines[0]
        self.assertAlmostEqual(first.start_sec, 1.0)
        self.assertAlmostEqual(first.end_sec, 2.3)
        self.assertEqual([word.text for word in first.words], ["月", "亮", "代"])
        self.assertAlmostEqual(document.lines[1].end_sec, 4.5)

    def test_zero_duration_words_get_a_minimum(self):
        document = lyrics.parse_qrc("[0,500]啊(0,0)\n")

        word = document.lines[0].words[0]
        self.assertGreater(word.end_sec, word.start_sec)


class AlignJsonTests(unittest.TestCase):
    def test_word_level_entries_keep_their_language_and_reading(self):
        payload = {
            "version": 1,
            "language": "zh-CN",
            "audio_duration_sec": 12.5,
            "lines": [
                {
                    "lang": "ja-JP",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "愛してる",
                    "words": [
                        {"text": "愛", "start": 1.0, "end": 1.5, "reading": "あい"},
                        {
                            "text": "してる",
                            "start": 1.5,
                            "end": 2.0,
                            "lang": "ja",
                        },
                    ],
                }
            ],
        }

        document = lyrics.parse_align_json(payload)
        words = lyrics.to_word_segments(document)

        self.assertEqual(len(words), 2)
        self.assertEqual(words[0].language_code, "ja-JP")
        self.assertEqual(words[1].text, "してる")
        self.assertAlmostEqual(document.audio_duration_sec, 12.5)

    def test_flat_word_list_is_accepted(self):
        document = lyrics.parse_align_json(
            {
                "version": 1,
                "words": [
                    {"text": "hello", "start": 0.0, "end": 0.5, "lang": "en"},
                    {"text": "世界", "start": 0.5, "end": 1.0, "lang": "zh"},
                ],
            }
        )
        words = lyrics.to_word_segments(document)

        self.assertEqual([w.language_code for w in words], ["en-US", "zh-CN"])

    def test_future_versions_are_rejected(self):
        with self.assertRaises(lyrics.LyricsError):
            lyrics.parse_align_json({"version": 2, "lines": []})


class WordSegmentTests(unittest.TestCase):
    def test_untimed_words_are_split_and_allocated(self):
        document = lyrics.parse_lrc(
            "[00:00.00]月亮\n[00:02.00]Sun\n",
            default_language_code="zh-CN",
            audio_duration_sec=4.0,
        )
        words = lyrics.to_word_segments(document)

        self.assertEqual([w.text for w in words], ["月", "亮", "Sun"])
        self.assertEqual(words[-1].language_code, "en-US")
        for word in words:
            self.assertGreater(word.end_sec, word.start_sec)
            self.assertEqual(word.confidence, 1.0)

    def test_empty_document_is_rejected(self):
        with self.assertRaises(lyrics.LyricsError):
            lyrics.parse_lrc("[ti:only metadata]\n")


class PipelineTests(unittest.TestCase):
    def test_stub_phonemizer_produces_bilabial_closure(self):
        words = [
            WordSegment("ma", 0.0, 0.4, language_code="zh-CN"),
            WordSegment("la", 0.4, 0.8, language_code="zh-CN"),
        ]

        document = lyrics.run_lyrics_pipeline(words, phonemizer=stub_phonemize)

        self.assertEqual(document.backend_id, "lyrics")
        self.assertEqual(document.model_id, "lyrics")
        self.assertEqual(document.events[0].viseme_id, "CLOSED")
        self.assertEqual(len(document.language_segments), 1)
        self.assertAlmostEqual(document.audio_duration_sec, 0.8)

    def test_mixed_languages_report_combined_document_language(self):
        words = [
            WordSegment("你", 0.0, 0.3, language_code="zh-CN"),
            WordSegment("hello", 0.3, 0.8, language_code="en-US"),
        ]

        document = lyrics.run_lyrics_pipeline(words, phonemizer=stub_phonemize)

        self.assertEqual(document.language_code, "mixed")
        self.assertEqual(
            [segment.language_code for segment in document.language_segments],
            ["zh-CN", "en-US"],
        )

    def test_empty_words_are_rejected(self):
        with self.assertRaises(lyrics.LyricsError):
            lyrics.run_lyrics_pipeline([], phonemizer=stub_phonemize)

    def test_default_language_is_inferred_from_words(self):
        words = [WordSegment("hello", 0.0, 0.5, language_code="en-US")]

        document = lyrics.run_lyrics_pipeline(words, phonemizer=stub_phonemize)

        self.assertEqual(document.language_code, "en-US")

    @unittest.skipUnless(HAS_PYPINYIN, "pypinyin is not installed")
    def test_real_chinese_g2p_produces_events(self):
        document = lyrics.parse_lrc(
            "[00:00.00]你好世界\n",
            default_language_code="zh-CN",
            audio_duration_sec=2.0,
        )
        words = lyrics.to_word_segments(document)

        result = lyrics.run_lyrics_pipeline(words)

        self.assertTrue(result.events)
        self.assertTrue(result.phonemes)


class ResolveWordsTests(unittest.TestCase):
    def test_lines_payload(self):
        words = lyrics.resolve_words(
            {
                "default_language_code": "zh-CN",
                "lines": [{"lang": "zh-CN", "start": 0.0, "end": 1.0, "text": "你好"}],
            }
        )

        self.assertEqual([w.text for w in words], ["你", "好"])

    def test_words_payload(self):
        words = lyrics.resolve_words(
            {
                "words": [{"text": "hi", "start": 0.0, "end": 0.4, "lang": "en"}],
            }
        )

        self.assertEqual(words[0].language_code, "en-US")

    def test_missing_input_is_rejected(self):
        with self.assertRaises(lyrics.LyricsError):
            lyrics.resolve_words({})


class FileLoadingTests(unittest.TestCase):
    def test_lrc_file_is_loaded_as_utf8_sig(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.lrc"
            path.write_text("\ufeff[00:01.00]你好\n", encoding="utf-8")

            document = lyrics.load_lyrics(str(path), default_language_code="zh-CN")

        self.assertEqual(document.lines[0].text, "你好")

    def test_json_file_is_loaded(self):
        payload = {
            "version": 1,
            "lines": [{"lang": "en-US", "start": 0.0, "end": 1.0, "text": "hello"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            document = lyrics.load_lyrics(str(path))

        self.assertEqual(document.source_format, "json")

    def test_missing_file_is_rejected(self):
        with self.assertRaises(lyrics.LyricsError):
            lyrics.load_lyrics("does-not-exist.lrc")


if __name__ == "__main__":
    unittest.main()
