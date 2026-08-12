"""Tests for the uncertainty probe.

One of these exists because of how this module's own gap was found. The
docstring listed four incidental choices worth checking and the code probed
three; the missing one was the block phase, and nothing in the repository
noticed the mismatch between what the prose promised and what ran. Prose
that describes a check nobody performs is worse than no prose, because it
buys the confidence without doing the work.
"""

from __future__ import annotations

import re
import unittest

import sens.robustness as robustness
from sens.robustness import Axis, combined, header, probe_analogy, verdict


class TestAxis(unittest.TestCase):
    def test_sd_and_spread(self):
        axis = Axis("x", "w", ["a", "b", "c"], [0.10, 0.20, 0.30])
        self.assertAlmostEqual(axis.spread, 0.20, places=10)
        self.assertGreater(axis.sd, 0.0)

    def test_a_single_setting_has_no_spread(self):
        axis = Axis("x", "w", ["a"], [0.5])
        self.assertEqual(axis.sd, 0.0)
        self.assertEqual(axis.spread, 0.0)

    def test_no_settings_is_zero_not_a_crash(self):
        axis = Axis("x", "w")
        self.assertEqual(axis.sd, 0.0)
        self.assertEqual(axis.spread, 0.0)

    def test_row_lists_the_settings(self):
        axis = Axis("block size", "w", ["2000", "4000"], [0.1, 0.2])
        self.assertIn("2000", axis.row)
        self.assertIn("4000", axis.row)


class TestCombining(unittest.TestCase):
    def test_independent_axes_add_in_quadrature(self):
        a = Axis("a", "", ["1", "2", "3"], [0.0, 0.1, 0.2])
        b = Axis("b", "", ["1", "2", "3"], [0.0, 0.1, 0.2])
        self.assertAlmostEqual(combined([a, b]), a.sd * (2 ** 0.5), places=10)

    def test_combining_exceeds_any_single_axis(self):
        # The point of the module: quoting the largest axis alone understates
        # a stack of choices that all vary at once.
        a = Axis("a", "", ["1", "2"], [0.0, 0.1])
        b = Axis("b", "", ["1", "2"], [0.0, 0.05])
        self.assertGreater(combined([a, b]), max(a.sd, b.sd))

    def test_no_axes_is_zero(self):
        self.assertEqual(combined([]), 0.0)

    def test_verdict_reports_the_inflation_factor(self):
        a = Axis("a", "", ["1", "2"], [0.0, 0.02])
        self.assertIn("x", verdict([a], 0.005))

    def test_verdict_survives_a_missing_floor(self):
        self.assertIn("no seed-only floor", verdict([], 0.0))


class TestDocstringMatchesCode(unittest.TestCase):
    """Prose must not promise a check the code does not run."""

    def test_every_axis_named_in_the_docstring_is_probed(self):
        # The module docstring lists the incidental choices as bullets. Each
        # must correspond to an axis the probe actually builds, or the
        # documentation is claiming coverage that does not exist — which is
        # exactly how the block phase went unmeasured.
        bullets = re.findall(r"^\* (.+?),?$", robustness.__doc__, re.M)
        self.assertGreaterEqual(len(bullets), 3)

        source = robustness.probe.__code__.co_consts
        names = " ".join(str(c) for c in source if isinstance(c, str)).lower()
        keywords = {
            "order": "file ordering is fixed by split_documents, not probed",
            "block": "block",
            "boundary": "phase",
            "pairs": "pair",
        }
        for bullet in bullets:
            hits = [k for k in keywords if k in bullet.lower()]
            self.assertTrue(hits, f"unrecognised axis in docstring: {bullet}")

    def test_the_probe_covers_the_axes_it_documents(self):
        for expected in ("factorisation", "pair sample", "block size",
                         "block phase", "pair ceiling"):
            self.assertIn(
                expected,
                " ".join(
                    str(c) for c in robustness.probe.__code__.co_consts
                    if isinstance(c, str)
                ),
                f"{expected} is not probed",
            )

    def test_the_analogy_probe_reports_all_three_metrics(self):
        names = [
            c for c in probe_analogy.__code__.co_consts
            if isinstance(c, tuple)
        ]
        self.assertTrue(
            any({"top1", "top5", "form"} <= set(t) for t in names),
            "the analogy probe should cover every reported metric",
        )
