import unittest

from mmd_mouth.core.schema import WordSegment
from mmd_mouth.language.blend import (
    BLENDS,
    blend_for_syllable,
    build_blended_events,
    split_syllable,
)


class SyllableSplitTests(unittest.TestCase):
    def test_w_spelling_is_read_as_a_u_medial(self):
        self.assertEqual(split_syllable("wo"), ("w", "uo"))
        self.assertEqual(split_syllable("wei"), ("w", "ui"))

    def test_y_spelling_is_read_as_an_i_medial(self):
        self.assertEqual(split_syllable("yan"), ("y", "ian"))
        self.assertEqual(split_syllable("yao"), ("y", "iao"))

    def test_umlaut_is_normalised(self):
        self.assertEqual(split_syllable("yue"), ("y", "ve"))
        self.assertEqual(split_syllable("lv"), ("l", "v"))

    def test_consonant_only_syllable_keeps_its_initial(self):
        self.assertEqual(split_syllable("zhi"), ("zh", "i"))


class BlendWeightTests(unittest.TestCase):
    def test_weights_stay_within_one_opening(self):
        # Nasal finals (an/en/in…) deliberately sum below 1.0: the mouth opens
        # less than for an oral vowel, so only the upper bound is strict.
        for final, weights in BLENDS.items():
            total = sum(weights.values())
            self.assertLessEqual(total, 1.001, msg=final)
            self.assertGreaterEqual(total, 0.84, msg=final)

    def test_diphthongs_blend_two_channels(self):
        _initial, final, weights = blend_for_syllable("ai")
        self.assertEqual(final, "ai")
        self.assertIn("A", weights)
        self.assertIn("I", weights)

    def test_apical_i_is_not_a_pure_i(self):
        _initial, _final, weights = blend_for_syllable("shi")
        self.assertLess(weights["I"], 1.0)
        self.assertIn("E", weights)

    def test_unknown_final_falls_back(self):
        _initial, _final, weights = blend_for_syllable("zzz")
        self.assertTrue(weights)


class BlendedEventTests(unittest.TestCase):
    def test_a_diphthong_produces_simultaneous_events(self):
        words = [WordSegment(text="爱", start_sec=0.0, end_sec=0.5, language_code="zh-CN")]

        events = build_blended_events(words, default_language_code="zh-CN")

        vowels = [event for event in events if event.viseme_id in {"A", "I", "U", "E", "O"}]
        self.assertEqual(len(vowels), 2)
        self.assertEqual({event.viseme_id for event in vowels}, {"A", "I"})
        self.assertAlmostEqual(sum(event.weight for event in vowels), 1.0, places=2)

    def test_bilabial_initial_adds_a_closure(self):
        words = [WordSegment(text="我", start_sec=0.0, end_sec=0.5, language_code="zh-CN")]
        events = build_blended_events(words, default_language_code="zh-CN")
        self.assertFalse(any(event.viseme_id == "CLOSED" for event in events))

        words = [WordSegment(text="不", start_sec=0.0, end_sec=0.5, language_code="zh-CN")]
        events = build_blended_events(words, default_language_code="zh-CN")
        self.assertTrue(any(event.viseme_id == "CLOSED" for event in events))

    def test_non_chinese_words_fall_back_to_g2p(self):
        words = [WordSegment(text="hello", start_sec=0.0, end_sec=0.5, language_code="en-US")]
        events = build_blended_events(words, default_language_code="en-US")
        self.assertTrue(events)
        self.assertTrue(all(event.viseme_id in {"A", "I", "U", "E", "O", "REST", "CLOSED"} for event in events))


if __name__ == "__main__":
    unittest.main()
