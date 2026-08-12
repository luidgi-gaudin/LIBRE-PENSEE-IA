"""Tests for the noise-floor estimator.

Nothing subtle to verify numerically — the point of the module is the
bookkeeping, so these check that the bookkeeping is right and that the
verdict function draws its lines where it says it does.
"""

from __future__ import annotations

import unittest

from sens.noise import DEFAULT_SEEDS, Spread, measure, verdict
from sens.pipeline import Config


class TestSpread(unittest.TestCase):
    def setUp(self):
        self.spread = Spread("held-out", percent=False,
                             values=[0.60, 0.61, 0.59, 0.60])

    def test_mean(self):
        self.assertAlmostEqual(self.spread.mean, 0.60, places=10)

    def test_noise_floor_is_two_standard_deviations(self):
        self.assertAlmostEqual(self.spread.floor, 2 * self.spread.sd, places=12)

    def test_a_single_measurement_has_no_spread(self):
        lonely = Spread("x", percent=False, values=[0.5])
        self.assertEqual(lonely.sd, 0.0)
        self.assertEqual(lonely.floor, 0.0)

    def test_no_measurements_is_zero_not_a_crash(self):
        empty = Spread("x", percent=False)
        self.assertEqual(empty.mean, 0.0)
        self.assertEqual(empty.sd, 0.0)

    def test_percent_formatting(self):
        self.assertEqual(Spread("t", percent=True).format(0.042), "4.2%")

    def test_plain_formatting(self):
        self.assertEqual(Spread("t", percent=False).format(0.6002), "0.6002")

    def test_row_reports_the_floor(self):
        self.assertIn("noise floor", self.spread.row)

    def test_row_reports_the_range(self):
        self.assertIn("0.5900-0.6100", self.spread.row)


class TestVerdict(unittest.TestCase):
    def setUp(self):
        # Sample standard deviation of [-1, 0, 1] is exactly 1.0, which
        # makes the sigma arithmetic below readable.
        self.spread = Spread("x", percent=False, values=[-1.0, 0.0, 1.0])

    def test_small_differences_are_noise(self):
        self.assertIn("within noise", verdict(self.spread, 1.0))

    def test_two_to_three_sigma_is_marginal(self):
        self.assertIn("marginal", verdict(self.spread, 2.5))

    def test_beyond_three_sigma_is_solid(self):
        self.assertIn("solid", verdict(self.spread, 4.0))

    def test_the_sign_of_the_difference_does_not_matter(self):
        self.assertEqual(verdict(self.spread, 4.0), verdict(self.spread, -4.0))

    def test_no_spread_is_reported_rather_than_dividing_by_zero(self):
        flat = Spread("x", percent=False, values=[1.0])
        self.assertIn("no spread", verdict(flat, 0.5))


def benchmarkable():
    """A tiny space the benchmark generator can actually find pairs in."""
    from sens.space import Space

    stems = ["ship", "whale", "boat", "sail", "rope", "mast"]
    words, vectors = [], []
    for i, stem in enumerate(stems):
        base = [0.0] * (len(stems) + 1)
        base[i] = 1.0
        words += [stem, stem + "s"]
        vectors.append(base)
        vectors.append([*base[:-1], 1.0])
    return Space(words=words, vectors=vectors,
                 meta={"eigenvalues": [1.0] * (len(stems) + 1), "config": {}})


class RecordingRuler:
    def __init__(self):
        self.seen = []

    def score(self, config: Config) -> float:
        self.seen.append(config.seed)
        return 0.6 + config.seed % 3 * 0.001


class TestMeasure(unittest.TestCase):
    def test_default_seeds_are_distinct(self):
        self.assertEqual(len(DEFAULT_SEEDS), len(set(DEFAULT_SEEDS)))

    def test_enough_seeds_for_a_standard_deviation(self):
        self.assertGreaterEqual(len(DEFAULT_SEEDS), 3)

    def test_it_varies_only_the_seed(self):
        # The whole estimate is worthless if anything else moves between
        # runs, so this pins that only `seed` differs from the defaults.
        seen = []
        base = Config().as_dict()

        class Spy(RecordingRuler):
            def score(self, config: Config) -> float:
                seen.append(config.as_dict())
                return 0.6

        import sens.noise as module

        original = module.build
        module.build = lambda paths, config, verbose=False: (benchmarkable(), None)
        try:
            module.measure([], Spy(), seeds=(1, 2), limit=4)
        finally:
            module.build = original

        for config in seen:
            differing = [k for k in base if base[k] != config[k]]
            self.assertEqual(differing, ["seed"], differing)

    def test_one_spread_per_metric(self):
        import sens.noise as module

        original = module.build
        module.build = lambda paths, config, verbose=False: (benchmarkable(), None)
        try:
            spreads = module.measure([], RecordingRuler(), seeds=(1, 2), limit=4)
        finally:
            module.build = original
        self.assertEqual(
            [s.metric for s in spreads], ["held-out", "top1", "top5", "form"]
        )
        for spread in spreads:
            self.assertEqual(len(spread.values), 2)


class TestUnmeasurableCorpus(unittest.TestCase):
    def test_a_vocabulary_with_no_benchmark_says_so(self):
        # Found by a test: this used to die on a bare KeyError from the
        # totals dict, which tells the caller nothing about what went wrong.
        from sens.space import Space
        import sens.noise as module

        empty = Space(words=["a", "b"], vectors=[[1.0], [2.0]],
                      meta={"eigenvalues": [1.0], "config": {}})
        original = module.build
        module.build = lambda paths, config, verbose=False: (empty, None)
        try:
            with self.assertRaises(ValueError) as caught:
                module.measure([], RecordingRuler(), seeds=(1,), limit=2)
        finally:
            module.build = original
        self.assertIn("no benchmark questions", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
