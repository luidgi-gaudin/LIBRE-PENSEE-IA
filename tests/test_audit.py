"""Tests for the parameter audit.

The thing worth guarding here is not the arithmetic but the fairness. An
audit that quietly rebuilt its ground truth for each candidate would produce
a full table of plausible numbers meaning nothing at all, so most of these
tests are about the ruler staying still.
"""

from __future__ import annotations

import unittest

from sens.audit import DEFAULT_GRID, Finding, Ruler, audit, build_ruler
from sens.linalg import SparseMatrix
from sens.pipeline import Config
from sens.text import Vocabulary


class TestFinding(unittest.TestCase):
    def test_best_picks_the_highest_score(self):
        finding = Finding("window", 4, [(2, 0.10), (4, 0.30), (6, 0.20)])
        self.assertEqual(finding.best, (4, 0.30))

    def test_default_wins_when_it_is_the_best(self):
        self.assertTrue(Finding("window", 4, [(2, 0.1), (4, 0.3)]).default_wins)

    def test_default_loses_when_something_beats_it(self):
        self.assertFalse(Finding("window", 2, [(2, 0.1), (4, 0.3)]).default_wins)

    def test_row_flags_a_losing_default(self):
        row = Finding("alpha", 0.75, [(0.75, 0.1), (1.0, 0.3)]).row
        self.assertIn("1.0 beats 0.75", row)

    def test_row_stays_quiet_when_the_default_wins(self):
        row = Finding("alpha", 1.0, [(0.75, 0.1), (1.0, 0.3)]).row
        self.assertNotIn("beats", row)

    def test_row_lists_every_value(self):
        row = Finding("shift", 1.0, [(1.0, 0.5), (2.0, 0.4), (5.0, 0.3)]).row
        for value in ("1.0=", "2.0=", "5.0="):
            self.assertIn(value, row)


class TestGrid(unittest.TestCase):
    def test_grid_names_are_real_config_fields(self):
        config = Config()
        for parameter, _ in DEFAULT_GRID:
            self.assertTrue(hasattr(config, parameter), parameter)

    def test_every_default_appears_in_its_own_grid(self):
        # If a default is missing from its grid the audit cannot tell you
        # whether it won, only that something scored well.
        config = Config()
        for parameter, values in DEFAULT_GRID:
            self.assertIn(getattr(config, parameter), values, parameter)

    def test_vocabulary_parameters_are_excluded(self):
        # Changing these changes which words exist, so no fixed ruler
        # survives and the comparison would be meaningless.
        names = {p for p, _ in DEFAULT_GRID}
        self.assertNotIn("vocab_size", names)
        self.assertNotIn("min_count", names)


class RecordingRuler(Ruler):
    """A ruler that records what it was asked to score instead of building."""

    def __init__(self, responses):
        self.responses = responses
        self.seen = []

    def score(self, config: Config) -> float:
        self.seen.append(config.as_dict())
        key = (config.window, config.alpha, config.shift,
               config.min_pair_weight, config.eigenvalue_power)
        return self.responses.get(key, 0.0)


class TestAudit(unittest.TestCase):
    def setUp(self):
        base = Config()
        self.base = base
        self.default_key = (base.window, base.alpha, base.shift,
                            base.min_pair_weight, base.eigenvalue_power)

    def test_returns_one_finding_per_parameter(self):
        ruler = RecordingRuler({self.default_key: 0.5})
        findings = audit(ruler, base=self.base)
        self.assertEqual(
            [f.parameter for f in findings], [p for p, _ in DEFAULT_GRID]
        )

    def test_each_finding_covers_its_whole_grid(self):
        ruler = RecordingRuler({})
        findings = audit(ruler, base=self.base)
        for finding, (_, values) in zip(findings, DEFAULT_GRID):
            self.assertEqual([v for v, _ in finding.scores], list(values))

    def test_the_default_build_is_scored_only_once(self):
        # The default value appears in all five grids and produces an
        # identical build every time. Rebuilding it five times would waste
        # minutes for nothing.
        ruler = RecordingRuler({self.default_key: 0.5})
        audit(ruler, base=self.base)
        defaults = [c for c in ruler.seen if (
            c["window"], c["alpha"], c["shift"],
            c["min_pair_weight"], c["eigenvalue_power"]
        ) == self.default_key]
        self.assertEqual(len(defaults), 1)

    def test_only_one_parameter_moves_at_a_time(self):
        ruler = RecordingRuler({})
        audit(ruler, base=self.base)
        reference = self.base.as_dict()
        for seen in ruler.seen:
            differing = [
                k for k in reference
                if reference[k] != seen[k]
            ]
            self.assertLessEqual(len(differing), 1, differing)

    def test_a_winning_alternative_is_detected(self):
        ruler = RecordingRuler({
            self.default_key: 0.50,
            (10, self.base.alpha, self.base.shift,
             self.base.min_pair_weight, self.base.eigenvalue_power): 0.90,
        })
        findings = {f.parameter: f for f in audit(ruler, base=self.base)}
        self.assertFalse(findings["window"].default_wins)
        self.assertEqual(findings["window"].best[0], 10)

    def test_progress_is_reported_for_each_build(self):
        calls = []
        ruler = RecordingRuler({})
        audit(ruler, base=self.base,
              progress=lambda p, v: calls.append((p, v)))
        self.assertEqual(len(calls), len(ruler.seen))

    def test_a_custom_grid_is_honoured(self):
        ruler = RecordingRuler({})
        findings = audit(ruler, grid=(("shift", (1.0, 3.0)),), base=self.base)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].parameter, "shift")


class TestBuildRuler(unittest.TestCase):
    def test_ruler_holds_disjoint_halves(self):
        import os
        import tempfile

        text = " ".join(["ship sails the wide salt sea"] * 400)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            ruler = build_ruler(
                [path],
                base=Config(vocab_size=20, min_count=1),
                block=200,
                pair_vocab=10,
                pair_count=5,
            )
        self.assertGreater(len(ruler.train), 0)
        self.assertGreater(len(ruler.test), 0)
        self.assertIsInstance(ruler.vocab, Vocabulary)
        self.assertIsInstance(ruler.truth, SparseMatrix)

    def test_pairs_are_ordered_and_distinct(self):
        import os
        import tempfile

        text = " ".join(["alpha beta gamma delta epsilon zeta"] * 500)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            ruler = build_ruler(
                [path],
                base=Config(vocab_size=10, min_count=1),
                block=120,
                pair_vocab=6,
                pair_count=10,
            )
        self.assertEqual(ruler.pairs, sorted(set(ruler.pairs)))
        for i, j in ruler.pairs:
            self.assertLess(i, j)


if __name__ == "__main__":
    unittest.main()
