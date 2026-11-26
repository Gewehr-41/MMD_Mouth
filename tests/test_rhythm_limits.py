import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import rhythm_vmd as core  # noqa: E402
from mmd_mouth.core.schema import VisemeEvent  # noqa: E402


class ClampTests(unittest.TestCase):
    def test_scale_then_clamp(self):
        self.assertAlmostEqual(core.clamp_channel(1.0, scale=0.75, low=0.0, high=0.95), 0.75)
        self.assertAlmostEqual(core.clamp_channel(1.0, scale=1.0, low=0.0, high=0.6), 0.6)

    def test_lower_bound_applies(self):
        self.assertAlmostEqual(core.clamp_channel(0.02, scale=1.0, low=0.05, high=1.0), 0.05)

    def test_reversed_bounds_are_swapped(self):
        self.assertAlmostEqual(core.clamp_channel(0.9, scale=1.0, low=0.8, high=0.2), 0.8)

    def test_bounds_are_sanitised(self):
        self.assertAlmostEqual(core.clamp_channel(0.5, scale=1.0, low=-1.0, high=2.0), 0.5)


class SampleLimitTests(unittest.TestCase):
    def setUp(self):
        self.events = [VisemeEvent("A", 0.0, 0.5, weight=1.0, priority=50)]

    def test_upper_limit_caps_the_curve(self):
        curves = core.sample_curves(
            self.events,
            1.0,
            fps=30.0,
            limits={"A": (0.0, 0.6)},
        )
        self.assertLessEqual(max(value for _frame, value in curves["あ"]), 0.6)

    def test_scale_shrinks_without_flattening(self):
        curves = core.sample_curves(self.events, 1.0, fps=30.0, scale=0.5)
        values = [value for _frame, value in curves["あ"]]
        self.assertLessEqual(max(values), 0.5)
        # The shape is preserved: it still varies over the syllable.
        self.assertGreater(max(values), min(values))

    def test_default_limits_keep_full_range(self):
        curves = core.sample_curves(self.events, 1.0, fps=30.0)
        self.assertAlmostEqual(max(value for _frame, value in curves["あ"]), 1.0, places=2)


class VariantTests(unittest.TestCase):
    def test_default_names_are_plain_vowels(self):
        self.assertEqual(
            [core.channel_morph_name(channel) for channel in core.CHANNELS],
            ["あ", "い", "う", "え", "お"],
        )

    def test_variant_suffixes_use_full_width_digits(self):
        self.assertEqual(core.channel_morph_name("A", 2), "あ２")
        self.assertEqual(core.channel_morph_name("U", 3), "う３")

    def test_invalid_variants_fall_back_to_one(self):
        self.assertEqual(core.normalize_variant(0), 1)
        self.assertEqual(core.normalize_variant(9), 3)
        self.assertEqual(core.normalize_variant("x"), 1)

    def test_sampling_renames_the_output_channel(self):
        events = [VisemeEvent("U", 0.0, 0.5, weight=1.0, priority=50)]

        default = core.sample_curves(events, 1.0, fps=30.0)
        variant = core.sample_curves(events, 1.0, fps=30.0, variants={"U": 2})

        self.assertIn("う", default)
        self.assertIn("う２", variant)
        self.assertNotIn("う", variant)


if __name__ == "__main__":
    unittest.main()
