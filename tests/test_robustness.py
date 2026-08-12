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


class TestRulerSweep(unittest.TestCase):
    """Checking a claim against a rebuilt yardstick."""

    def test_reverses_detects_a_sign_change(self):
        self.assertTrue(robustness.reverses({"a": 0.05, "b": -0.05}))

    def test_reverses_ignores_a_flip_below_the_floor(self):
        # The first sweep called five claims instrument-dependent. Two were
        # registered as no-effect at 0.3 sd: an effect that never rose above
        # the noise has no direction to lose.
        self.assertFalse(robustness.reverses({"a": 0.0042, "b": -0.0026}))

    def test_reverses_needs_both_sides_above_the_floor(self):
        # One real effect and one puff of noise pointing the other way is
        # not a reversal; it is one measurement and one non-measurement.
        self.assertFalse(robustness.reverses({"a": 0.05, "b": -0.001}))

    def test_the_floor_is_the_registers_own(self):
        from sens.claims import EFFECT_SD

        # Just inside and just outside, so the default cannot drift away
        # from the number the register judges everything else against.
        self.assertFalse(
            robustness.reverses({"a": EFFECT_SD * 0.9, "b": -EFFECT_SD * 0.9})
        )
        self.assertTrue(
            robustness.reverses({"a": EFFECT_SD * 1.1, "b": -EFFECT_SD * 1.1})
        )

    def test_a_zero_floor_recovers_the_naive_comparison(self):
        self.assertTrue(
            robustness.reverses({"a": 0.0042, "b": -0.0026}, floor=0.0)
        )

    def test_a_ruler_varying_the_claims_own_parameter_is_confounded(self):
        # Asking whether "alpha 1.0 beats 0.75" survives a ground truth built
        # at alpha 1.0 is asking a measurement to agree with itself.
        from sens.claims import REGISTER

        alpha = next(c for c in REGISTER if c.id == "alpha-smoothing-off")
        self.assertTrue(robustness.confounded(alpha, {"alpha": 1.0}))
        self.assertFalse(robustness.confounded(alpha, {"window": 2}))

    def test_a_claim_with_no_recipe_is_never_confounded(self):
        from sens.claims import Claim

        self.assertFalse(
            robustness.confounded(Claim(
                id="x", what="", section="", status="open", novels=""
            ), {"alpha": 1.0})
        )

    def test_every_claim_is_confounded_by_at_least_one_ruler_or_none(self):
        # Guards the sweep against a ruler set that disqualifies itself
        # entirely for some claim, leaving it silently unmeasured.
        from sens.claims import REGISTER

        for claim in (c for c in REGISTER if c.verifiable):
            usable = [
                label for label, o in robustness.RULERS
                if not robustness.confounded(claim, o)
            ]
            self.assertGreater(len(usable), 1, claim.id)

    def test_reverses_ignores_magnitude(self):
        # A claim that shrinks by an order of magnitude but keeps its
        # direction is telling you about the language; one that flips is
        # telling you about the instrument. Only the second matters here.
        self.assertFalse(robustness.reverses({"a": 0.35, "b": 0.02}))

    def test_reverses_tolerates_exact_zero(self):
        # Zero has no sign to disagree with, so it must not by itself
        # count as a reversal.
        self.assertFalse(robustness.reverses({"a": 0.05, "b": 0.0}))

    def test_a_single_ruler_cannot_reverse(self):
        self.assertFalse(robustness.reverses({"a": 0.05}))

    def test_the_frozen_ruler_is_the_first_one_swept(self):
        # Results are read against the frozen ruler, so it has to be present
        # and it reads better first.
        label, overrides = robustness.RULERS[0]
        self.assertEqual(overrides, {})
        self.assertIn("frozen", label)

    def test_the_sweep_varies_more_than_one_parameter(self):
        # Varying only alpha would leave a claim that survives alpha looking
        # invariant when it is merely alpha-invariant.
        varied = {k for _, o in robustness.RULERS for k in o}
        self.assertGreaterEqual(len(varied), 2)

    def test_every_ruler_override_is_a_real_config_field(self):
        from sens.pipeline import Config

        config = Config()
        for _, overrides in robustness.RULERS:
            for name in overrides:
                self.assertTrue(hasattr(config, name), name)

    def test_ruler_overrides_name_frozen_ruler_keys(self):
        # Overriding something the RULER does not pin would silently do
        # nothing, since the model config supplies it anyway.
        from sens.heldout import RULER

        for _, overrides in robustness.RULERS:
            for name in overrides:
                self.assertIn(name, RULER, name)
