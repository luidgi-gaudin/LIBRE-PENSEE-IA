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
