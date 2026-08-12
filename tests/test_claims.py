"""Tests for the claims register.

The register exists because the controls that caught six wrong claims all
had to be remembered at the time. These tests are the part that does the
remembering: they refuse to let a claim be stated with more confidence than
its evidence, refuse to let the register drift from the README, and refuse
to let a new finding be added without saying what was done to check it.
"""

from __future__ import annotations

import os
import re
import unittest

from sens.claims import (
    CONTROLS,
    REGISTER,
    REQUIRED,
    STATUSES,
    Claim,
    by_status,
    unentitled,
    unreplicated,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRegisterShape(unittest.TestCase):
    def test_identifiers_are_unique(self):
        ids = [c.id for c in REGISTER]
        self.assertEqual(len(ids), len(set(ids)))

    def test_identifiers_are_slugs(self):
        for claim in REGISTER:
            self.assertRegex(claim.id, r"^[a-z0-9]+(-[a-z0-9]+)*$")

    def test_every_status_is_known(self):
        for claim in REGISTER:
            self.assertIn(claim.status, STATUSES, claim.id)

    def test_every_control_is_documented(self):
        for claim in REGISTER:
            for control in claim.controls:
                self.assertIn(control, CONTROLS, f"{claim.id}: {control}")

    def test_every_status_has_a_requirement_entry(self):
        for status in STATUSES:
            self.assertIn(status, REQUIRED)

    def test_by_status_partitions_the_register(self):
        total = sum(len(by_status(s)) for s in STATUSES)
        self.assertEqual(total, len(REGISTER))

    def test_the_register_is_not_trivially_small(self):
        self.assertGreater(len(REGISTER), 15)


class TestEntitlement(unittest.TestCase):
    """The check that would have caught the claim this repo most regrets."""

    def test_no_claim_outruns_its_controls(self):
        weak = [(c.id, sorted(c.missing)) for c in unentitled()]
        self.assertEqual(weak, [], f"unsupported claims: {weak}")

    def test_holds_requires_replication(self):
        # Stated confidence about the *method* needs more than one corpus.
        # `compression-trades-identity` had a noise floor, three agreeing
        # measurements and one corpus, and was wrong.
        self.assertIn("second-corpus", REQUIRED["holds"])
        for claim in by_status("holds"):
            self.assertTrue(claim.replicated, claim.id)

    def test_refutations_are_paired_and_replicated(self):
        for claim in by_status("refuted"):
            self.assertIn("second-corpus", claim.controls, claim.id)
            self.assertIn("paired-test", claim.controls, claim.id)

    def test_every_claim_records_a_noise_floor_or_is_open(self):
        for claim in REGISTER:
            if claim.status == "open":
                continue
            self.assertIn("noise-floor", claim.controls, claim.id)

    def test_replication_is_reported_honestly(self):
        # A claim carrying the second-corpus control must actually say what
        # the second corpus showed, rather than leaving the default.
        for claim in REGISTER:
            if claim.replicated:
                self.assertNotEqual(claim.expository, "not run", claim.id)

    def test_unreplicated_claims_admit_it(self):
        for claim in unreplicated():
            self.assertEqual(claim.expository, "not run", claim.id)


class TestAgainstTheReadme(unittest.TestCase):
    """The register and the prose must not drift apart."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as handle:
            cls.text = handle.read()
        cls.headings = {
            re.sub(r"\s+", "-",
                   re.sub(r"[^\w\s-]", "", line.lstrip("#").strip().lower()))
            for line in cls.text.splitlines() if line.startswith("#")
        }

    def test_every_claim_points_at_a_real_section(self):
        missing = sorted(
            {c.section for c in REGISTER if c.section not in self.headings}
        )
        self.assertEqual(missing, [], f"dangling sections: {missing}")

    def test_every_retracted_claim_is_still_discussed(self):
        # Retractions are the most useful part; they must not be quietly
        # dropped from the prose once they stop being news.
        for claim in by_status("refuted"):
            self.assertIn(claim.section, self.headings, claim.id)

    def test_the_open_question_is_not_silently_closed(self):
        self.assertTrue(by_status("open"),
                        "an unexplained result was removed rather than solved")


class TestClaimBehaviour(unittest.TestCase):
    def test_missing_lists_exactly_the_absent_controls(self):
        claim = Claim(id="x", what="w", section="s", status="holds",
                      novels="n", controls=frozenset({"noise-floor"}))
        self.assertEqual(claim.missing, frozenset({"second-corpus"}))
        self.assertFalse(claim.entitled)

    def test_a_fully_controlled_claim_is_entitled(self):
        claim = Claim(id="x", what="w", section="s", status="holds",
                      novels="n", expository="e",
                      controls=frozenset({"noise-floor", "second-corpus"}))
        self.assertTrue(claim.entitled)
        self.assertTrue(claim.replicated)

    def test_open_claims_require_nothing(self):
        claim = Claim(id="x", what="w", section="s", status="open", novels="n")
        self.assertTrue(claim.entitled)


if __name__ == "__main__":
    unittest.main()


class TestVerifiability(unittest.TestCase):
    """The half of the register that does not take its own word for it."""

    def test_a_useful_share_of_claims_can_be_re_derived(self):
        checkable = [c for c in REGISTER if c.verifiable]
        self.assertGreaterEqual(len(checkable), 8)

    def test_a_recipe_and_a_number_travel_together(self):
        # Either both or neither: a recipe with nothing to compare against
        # verifies nothing, and a number with no recipe cannot be checked.
        for claim in REGISTER:
            self.assertEqual(
                claim.check is not None,
                claim.effect is not None,
                claim.id,
            )

    def test_every_recipe_names_a_real_parameter(self):
        from sens.pipeline import Config

        config = Config()
        for claim in REGISTER:
            if claim.check:
                self.assertTrue(
                    hasattr(config, claim.check.parameter),
                    f"{claim.id}: {claim.check.parameter}",
                )

    def test_every_recipe_actually_changes_something(self):
        from sens.pipeline import Config

        config = Config()
        for claim in REGISTER:
            if claim.check:
                self.assertNotEqual(
                    getattr(config, claim.check.parameter),
                    claim.check.against,
                    f"{claim.id} compares the default against itself",
                )

    def test_every_numeric_effect_has_a_unit(self):
        for claim in REGISTER:
            if claim.effect is not None:
                self.assertTrue(claim.unit, claim.id)

    def test_the_prose_and_the_number_agree_in_sign(self):
        for claim in REGISTER:
            if claim.effect is None:
                continue
            stated = claim.novels.strip()
            if stated.startswith(("+", "-")):
                self.assertEqual(
                    stated[0] == "-", claim.effect < 0,
                    f"{claim.id}: prose says {stated[0]}, number is {claim.effect}",
                )

    def test_every_unit_has_a_tolerance(self):
        from sens.verify import TOLERANCE

        for claim in REGISTER:
            if claim.effect is not None:
                self.assertIn(claim.unit, TOLERANCE, claim.id)


class TestNoiseFloorDiscipline(unittest.TestCase):
    """The floor a claim is judged against must be the honest one."""

    def test_the_effect_floor_counts_more_than_the_seed(self):
        # It used to be the factorisation seed alone, at 0.0047, which
        # overstated every claim by 2.4x. The pair sample turned out to be
        # the largest source and had never been looked at.
        from sens.claims import EFFECT_SD
        from sens.verify import HELDOUT_SD

        self.assertGreater(EFFECT_SD, HELDOUT_SD)

    def test_verification_tolerance_is_a_different_quantity(self):
        # Verification holds every seed fixed and reproduces exactly, so its
        # margin exists to catch code drift. Judging a claim must not borrow
        # it, because it excludes the choices that actually vary.
        from sens.claims import EFFECT_SD
        from sens.verify import TOLERANCE

        self.assertNotAlmostEqual(TOLERANCE["spearman"], EFFECT_SD, places=4)

    def test_every_sigma_matches_its_effect_and_the_floor(self):
        from sens.claims import EFFECT_SD

        for claim in REGISTER:
            if claim.effect is None or claim.sigma is None:
                continue
            if claim.unit != "spearman":
                continue
            expected = abs(claim.effect) / EFFECT_SD
            self.assertAlmostEqual(
                claim.sigma, expected, delta=0.15,
                msg=f"{claim.id}: sigma {claim.sigma} but "
                    f"{claim.effect}/{EFFECT_SD} = {expected:.2f}",
            )

    def test_form_sigmas_match_the_form_floor(self):
        from sens.claims import FORM_SD

        for claim in REGISTER:
            if claim.unit != "form" or claim.sigma in (None, 0.0):
                continue
            stated = re.search(r"([\d.]+) form points", claim.novels)
            if not stated:
                continue
            expected = float(stated.group(1)) / 100.0 / FORM_SD
            self.assertAlmostEqual(
                claim.sigma, expected, delta=0.15,
                msg=f"{claim.id}: sigma {claim.sigma} but "
                    f"{stated.group(1)} pts / {FORM_SD} = {expected:.2f}",
            )

    def test_paired_claims_are_exempt_from_the_point_floor(self):
        # A McNemar test over one question set cannot be moved by which
        # questions were drawn. Every refutation here is paired, which is
        # why they survived the correction that halved several sigmas.
        for claim in by_status("refuted"):
            self.assertTrue(claim.paired, claim.id)

    def test_statuses_agree_with_their_sigmas(self):
        # Three standard deviations is where this repository draws `holds`.
        # An effect below it that still says `holds` is the exact mistake
        # the noise floor exists to prevent.
        for claim in REGISTER:
            if claim.sigma is None or claim.sigma == 0.0:
                continue
            if claim.paired:
                continue
            if claim.status == "holds":
                self.assertGreaterEqual(claim.sigma, 3.0, claim.id)
            if claim.status == "no-effect":
                self.assertLess(claim.sigma, 2.0, claim.id)


class TestRulerInvariance(unittest.TestCase):
    """The yardstick has parameters too, and one claim reverses under them."""

    def test_the_control_exists_and_is_documented(self):
        self.assertIn("ruler-invariant", CONTROLS)

    def test_instrument_dependence_is_a_status(self):
        self.assertIn("instrument-dependent", STATUSES)

    def test_an_instrument_dependent_claim_must_have_been_checked(self):
        # The status means "measured against a rebuilt ruler and reversed",
        # not "never looked at". Those are opposite situations and must not
        # share a label.
        from sens.claims import REQUIRED

        self.assertIn("ruler-invariant", REQUIRED["instrument-dependent"])
        for claim in by_status("instrument-dependent"):
            self.assertTrue(claim.ruler_checked, claim.id)

    def test_the_unchecked_list_is_reported_not_empty_by_construction(self):
        from sens.claims import unchecked_against_the_ruler

        loose = unchecked_against_the_ruler()
        # Every entry must be a held-out claim genuinely lacking the control,
        # so the report cannot be quietly satisfied by narrowing what counts.
        for claim in loose:
            self.assertEqual(claim.unit, "spearman", claim.id)
            self.assertFalse(claim.ruler_checked, claim.id)

    def test_checked_claims_say_what_the_other_rulers_gave(self):
        # A control is worth nothing if the numbers behind it are not
        # written down; that is the lesson of the expository column.
        for claim in REGISTER:
            if claim.ruler_checked:
                self.assertTrue(
                    any(ch.isdigit() for ch in claim.note),
                    f"{claim.id}: claims ruler-invariance without figures",
                )
