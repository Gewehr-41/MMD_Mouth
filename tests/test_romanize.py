import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmd_mouth.language.romanize import romanize_letters


class RomanizeTests(unittest.TestCase):
    def test_chinese_uses_pinyin_letters(self):
        self.assertEqual(romanize_letters("爱", "zh-CN"), ["a", "i"])
        self.assertEqual(romanize_letters("月亮", "zh-CN")[:2], ["y", "u"])

    def test_english_uses_the_spelling(self):
        self.assertEqual(romanize_letters("Baby", "en-US"), ["b", "a", "b", "y"])
        self.assertEqual(romanize_letters("hi!", "en-US"), ["h", "i"])

    def test_japanese_uses_openjtalk_romaji_phones(self):
        letters = romanize_letters("君の声が", "ja-JP")
        # 君 の 声 が -> k i m i | n o | k o e | g a
        self.assertEqual("".join(letters), "kiminokoega")

    def test_japanese_nasal_is_not_dropped(self):
        self.assertEqual("".join(romanize_letters("に変わる", "ja-JP")), "nikawaru")

    def test_simplified_character_falls_back_to_pinyin(self):
        letters = romanize_letters("爱", "ja-JP")
        self.assertTrue(letters)

    def test_unknown_language_uses_latin_letters(self):
        self.assertEqual(romanize_letters("ok", ""), ["o", "k"])


if __name__ == "__main__":
    unittest.main()
