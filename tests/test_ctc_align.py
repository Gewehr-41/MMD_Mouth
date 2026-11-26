import sys
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy lives in the worker venv
    HAS_NUMPY = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed")
class ViterbiTests(unittest.TestCase):
    def setUp(self):
        import ctc_align

        self.ctc = ctc_align

    def _emissions(self, frames, strong):
        logits = np.zeros((frames, 31), dtype=np.float32)
        for frame, token in strong:
            logits[frame, token] = 12.0
        logits[:, 0] += 6.0
        peak = logits.max(axis=-1, keepdims=True)
        return logits - peak - np.log(np.exp(logits - peak).sum(axis=-1, keepdims=True))

    def test_path_is_monotonic_and_hits_every_target(self):
        targets = [self.ctc.VOCAB[letter] for letter in "niwenwo"]
        emissions = self._emissions(
            60,
            [(5, targets[0]), (12, targets[1]), (20, targets[2]),
             (28, targets[3]), (36, targets[4]), (44, targets[5])],
        )
        path, _scores = self.ctc.viterbi(emissions, targets)

        self.assertEqual(len(path), 60)
        self.assertTrue(np.all(np.diff(path) >= 0), "path must never move backwards")
        indices = {int((state - 1) // 2) for state in path if state > 0}
        self.assertEqual(indices, set(range(len(targets))))

    def test_empty_targets_return_blank_path(self):
        emissions = self._emissions(10, [])
        path, _scores = self.ctc.viterbi(emissions, [])

        self.assertTrue(np.all(path == 0))


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed")
class CompactSpanTests(unittest.TestCase):
    def setUp(self):
        import ctc_align

        self.ctc = ctc_align

    def test_leading_gap_moves_start_to_the_tail(self):
        spans = [(0.0, 1.0), (1.0, 30.0), (30.0, 31.0)]

        result = self.ctc.compact_spans(spans)

        self.assertLess(result[1][1] - result[1][0], 3.0)
        self.assertAlmostEqual(result[1][1], 30.0)

    def test_trailing_gap_clamps_the_last_span(self):
        spans = [(0.0, 1.0), (1.0, 2.0), (2.0, 40.0)]

        result = self.ctc.compact_spans(spans)

        self.assertLess(result[-1][1] - result[-1][0], 3.0)
        self.assertAlmostEqual(result[-1][0], 2.0)

    def test_normal_spans_are_untouched(self):
        spans = [(0.0, 0.4), (0.4, 0.9), (0.9, 1.3)]

        self.assertEqual(self.ctc.compact_spans(spans), spans)


if __name__ == "__main__":
    unittest.main()
