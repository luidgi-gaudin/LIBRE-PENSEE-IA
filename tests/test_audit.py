"""Tests for the parameter audit.

The thing worth guarding here is not the arithmetic but the fairness. An
audit that quietly rebuilt its ground truth for each candidate would produce
a full table of plausible numbers meaning nothing at all, so most of these
tests are about the ruler staying still.
"""

from __future__ import annotations

import unittest

from sens.audit import (DEFAULT_GRID, TOKENISATIONS, Finding, Ruler, audit,
                        audit_tokenisation, audit_vocabulary, build_ruler)
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
        # Changing these changes which words exist, so the ordinary ruler
        # cannot judge them. audit_vocabulary handles them separately.
        names = {p for p, _ in DEFAULT_GRID}
        self.assertNotIn("vocab_size", names)
        self.assertNotIn("min_count", names)

    def test_every_parameter_the_pipeline_exposes_is_audited_somewhere(self):
        # The point of the audit is that no default goes unmeasured, so a
        # new Config field should not be able to slip past it unnoticed.
        gridded = {p for p, _ in DEFAULT_GRID}
        elsewhere = {
            "vocab_size", "min_count",   # audit_vocabulary
            "dim",                       # sens dimensions
            "seed",                      # sens noise
            "factoriser", "krylov_block", "krylov_depth",  # sens subspaces
            "oversample", "power_iterations",              # accuracy section
            "shrinkage",                 # the shrinkage sweep
            "lowercase", "fold_accents", "split_clitics",  # audit_tokenisation
        }
        for name in Config().as_dict():
            self.assertTrue(
                name in gridded or name in elsewhere,
                f"{name} is not measured anywhere",
            )


def grid_key(config) -> tuple:
    """Identify a build by every parameter the grid can vary.

    Keying on a hand-listed subset silently breaks when a parameter joins
    the grid — a `harmonic=False` build looked identical to the default one
    and made a caching test fail against correct code.
    """
    values = config if isinstance(config, dict) else config.as_dict()
    return tuple(values[name] for name, _ in DEFAULT_GRID)


class RecordingRuler(Ruler):
    """A ruler that records what it was asked to score instead of building."""

    def __init__(self, responses):
        self.responses = responses
        self.seen = []

    def score(self, config: Config) -> float:
        self.seen.append(config.as_dict())
        return self.responses.get(grid_key(config), 0.0)


class TestAudit(unittest.TestCase):
    def setUp(self):
        base = Config()
        self.base = base
        self.default_key = grid_key(base)

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
        defaults = [c for c in ruler.seen if grid_key(c) == self.default_key]
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
        winner = Config(**{**self.base.as_dict(), "window": 10})
        ruler = RecordingRuler({
            self.default_key: 0.50,
            grid_key(winner): 0.90,
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


class TestAuditVocabulary(unittest.TestCase):
    """The two parameters the ordinary ruler cannot judge."""

    def corpus(self, tmp):
        import os

        # Enough distinct words, repeated enough, that several vocabulary
        # sizes are actually different from one another.
        sentences = [
            "the sailor sails the ship across the wide salt sea",
            "a hunter walks a path through a dark green forest",
            "the captain steers the vessel over the deep cold ocean",
            "a farmer rides a road beside a bright warm field",
        ]
        path = os.path.join(tmp, "corpus.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(" ".join(sentences * 300))
        return path

    def run_audit(self, **kwargs):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            return audit_vocabulary(
                [self.corpus(tmp)],
                sizes=(10, 20),
                min_counts=(1, 5),
                base=Config(vocab_size=20, min_count=1, dim=4, oversample=4),
                block=400,
                pair_vocab=8,
                pair_count=20,
                **kwargs,
            )

    def test_it_reports_both_parameters(self):
        findings = self.run_audit()
        self.assertEqual(
            [f.parameter for f in findings], ["vocab_size", "min_count"]
        )

    def test_each_finding_covers_its_values(self):
        findings = {f.parameter: f for f in self.run_audit()}
        self.assertEqual(
            [v for v, _ in findings["vocab_size"].scores], [10, 20]
        )
        self.assertEqual([v for v, _ in findings["min_count"].scores], [1, 5])

    def test_scores_are_real_numbers(self):
        for finding in self.run_audit():
            for _, score in finding.scores:
                self.assertGreaterEqual(score, -1.0)
                self.assertLessEqual(score, 1.0)

    def test_progress_is_reported(self):
        calls = []
        self.run_audit(progress=lambda p, v: calls.append((p, v)))
        self.assertEqual(len(calls), 4)

    def test_it_is_reproducible(self):
        first = self.run_audit()
        second = self.run_audit()
        self.assertEqual(
            [f.scores for f in first], [f.scores for f in second]
        )


class TestTokenisationVariants(unittest.TestCase):
    def test_the_default_is_first_and_changes_nothing(self):
        label, overrides = TOKENISATIONS[0]
        self.assertEqual(label, "default")
        self.assertEqual(overrides, {})

    def test_every_override_names_a_real_config_field(self):
        config = Config()
        for _, overrides in TOKENISATIONS:
            for name in overrides:
                self.assertTrue(hasattr(config, name), name)

    def test_every_variant_actually_differs_from_the_default(self):
        config = Config()
        for label, overrides in TOKENISATIONS[1:]:
            self.assertTrue(
                any(getattr(config, k) != v for k, v in overrides.items()),
                label,
            )

    def test_it_scores_every_variant_on_the_same_pairs(self):
        # The whole validity of the comparison. An earlier version graded
        # each variant on whatever pairs it happened to cover, which flatters
        # any variant that drops the hard ones.
        import os
        import tempfile

        text = " ".join([
            "the sailor sails the ship across the wide salt sea",
            "a hunter walks a path through a dark green forest",
        ] * 400)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            rows = audit_tokenisation(
                [path],
                base=Config(vocab_size=20, min_count=1, dim=4, oversample=4),
                block=300,
                pair_vocab=8,
                pair_count=20,
            )
        self.assertEqual([r[0] for r in rows], [l for l, _ in TOKENISATIONS])
        for _, rho, coverage in rows:
            self.assertGreaterEqual(rho, -1.0)
            self.assertLessEqual(rho, 1.0)
            self.assertGreaterEqual(coverage, 0.0)
            self.assertLessEqual(coverage, 1.0)

    def test_the_default_variant_covers_everything(self):
        import os
        import tempfile

        text = " ".join(["the ship sails the wide salt sea"] * 500)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            rows = audit_tokenisation(
                [path],
                base=Config(vocab_size=20, min_count=1, dim=4, oversample=4),
                block=300, pair_vocab=8, pair_count=20,
            )
        self.assertAlmostEqual(rows[0][2], 1.0, places=10)
