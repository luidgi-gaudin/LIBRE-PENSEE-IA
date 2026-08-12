"""Tests for the uncompressed control and the paired significance test.

The significance test gets the most scrutiny. A p-value is the easiest thing
in the repository to compute confidently and wrongly, and unlike an accuracy
figure there is no way to eyeball that it is off.
"""

from __future__ import annotations

import math
import unittest

from sens.baseline import (
    Comparison,
    SparseSpace,
    Tally,
    compare,
    header,
    mcnemar,
)
from sens.evaluate import Category, Rule
from sens.linalg import SparseMatrix
from sens.space import Space


def sparse_space() -> SparseSpace:
    """Four words as sparse context rows.

    `alpha` and `beta` share their whole context; `gamma` and `delta` share
    a different one. So similarity within each pair should be 1 and across
    them 0, with no arithmetic left to interpretation.
    """
    return SparseSpace(
        ["alpha", "beta", "gamma", "delta"],
        SparseMatrix([
            {0: 3.0, 1: 4.0},
            {0: 6.0, 1: 8.0},
            {2: 1.0, 3: 1.0},
            {2: 2.0, 3: 2.0},
        ]),
    )


class TestSparseSpace(unittest.TestCase):
    def test_length_and_membership(self):
        s = sparse_space()
        self.assertEqual(len(s), 4)
        self.assertIn("alpha", s)
        self.assertNotIn("epsilon", s)

    def test_rows_are_normalised_at_construction(self):
        s = sparse_space()
        for row in s.units:
            if row:
                length = math.sqrt(sum(v * v for v in row.values()))
                self.assertAlmostEqual(length, 1.0, places=12)

    def test_parallel_rows_are_identical(self):
        self.assertAlmostEqual(sparse_space().similarity("alpha", "beta"), 1.0)

    def test_disjoint_rows_are_orthogonal(self):
        self.assertAlmostEqual(sparse_space().similarity("alpha", "gamma"), 0.0)

    def test_similarity_is_symmetric(self):
        s = sparse_space()
        self.assertAlmostEqual(
            s.similarity("alpha", "gamma"), s.similarity("gamma", "alpha")
        )

    def test_an_empty_row_survives_construction(self):
        s = SparseSpace(["a", "b"], SparseMatrix([{}, {0: 1.0}]))
        self.assertEqual(s.units[0], {})

    def test_dot_is_order_independent(self):
        # The implementation swaps to iterate the shorter row; that must not
        # be visible in the answer.
        small = {0: 1.0}
        large = {0: 1.0, 1: 1.0, 2: 1.0}
        self.assertAlmostEqual(
            SparseSpace._dot(small, large), SparseSpace._dot(large, small)
        )

    def test_analogy_excludes_its_own_inputs(self):
        s = sparse_space()
        returned = {w for w, _ in s.analogy("alpha", "beta", "gamma")}
        self.assertFalse(returned & {"alpha", "beta", "gamma"})

    def test_analogy_respects_the_count(self):
        self.assertEqual(
            len(sparse_space().analogy("alpha", "beta", "gamma", n=1)), 1
        )

    def test_analogy_returns_descending_scores(self):
        s = SparseSpace(
            ["a", "b", "c", "d", "e"],
            SparseMatrix([
                {0: 1.0}, {0: 1.0, 1: 1.0}, {2: 1.0},
                {2: 1.0, 1: 1.0}, {4: 1.0},
            ]),
        )
        scores = [x for _, x in s.analogy("a", "b", "c")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_analogy_solves_a_shared_offset(self):
        # a->b adds context 1; c->d adds the same context 1. So a:b::c:?
        # must return d.
        s = SparseSpace(
            ["a", "b", "c", "d", "e"],
            SparseMatrix([
                {0: 1.0}, {0: 1.0, 9: 1.0}, {2: 1.0},
                {2: 1.0, 9: 1.0}, {5: 1.0},
            ]),
        )
        self.assertEqual(s.analogy("a", "b", "c", n=1)[0][0], "d")

    def test_skips_words_with_empty_rows(self):
        s = SparseSpace(
            ["a", "b", "c", "blank"],
            SparseMatrix([{0: 1.0}, {0: 1.0, 1: 1.0}, {1: 1.0}, {}]),
        )
        self.assertNotIn("blank", [w for w, _ in s.analogy("a", "b", "c")])


class TestMcNemar(unittest.TestCase):
    def test_no_disagreements_is_no_evidence(self):
        self.assertEqual(mcnemar(0, 0), 1.0)

    def test_symmetric_in_its_arguments(self):
        self.assertAlmostEqual(mcnemar(57, 34), mcnemar(34, 57))

    def test_equal_disagreement_counts_give_a_large_p(self):
        # Not 1.0: with |b - c| = 0 the continuity correction still leaves
        # chi2 = 1/(b+c), so the p-value approaches 1 from below as the
        # counts grow rather than sitting on it.
        self.assertGreater(mcnemar(20, 20), 0.8)
        self.assertGreater(mcnemar(200, 200), mcnemar(20, 20))

    def test_lopsided_disagreement_gives_a_small_p(self):
        self.assertLess(mcnemar(30, 2), 0.001)

    def test_matches_the_chi_square_tail(self):
        # chi2 = (|57-34| - 1)^2 / 91 = 484/91 = 5.3187, whose one-degree-of-
        # freedom two-sided tail is 0.0211. These are the counts the raw vs
        # compressed top5 comparison actually produced.
        self.assertAlmostEqual(mcnemar(57, 34), 0.0211, places=4)

    def test_matches_a_second_hand_computed_tail(self):
        # chi2 = (|66-105| - 1)^2 / 171 = 1444/171 = 8.4444 -> 0.0037
        self.assertAlmostEqual(mcnemar(66, 105), 0.0037, places=4)

    def test_more_evidence_lowers_p_for_the_same_ratio(self):
        self.assertLess(mcnemar(60, 30), mcnemar(20, 10))

    def test_p_stays_in_range(self):
        for a in (0, 1, 5, 50):
            for b in (0, 1, 5, 50):
                self.assertGreaterEqual(mcnemar(a, b), 0.0)
                self.assertLessEqual(mcnemar(a, b), 1.0)


class TestTally(unittest.TestCase):
    def test_rates(self):
        tally = Tally("x", asked=200, top1=10, top5=40, form=50)
        self.assertAlmostEqual(tally.rate("top1"), 0.05)
        self.assertAlmostEqual(tally.rate("form"), 0.25)

    def test_no_questions_is_zero_not_a_crash(self):
        self.assertEqual(Tally("x").rate("top1"), 0.0)

    def test_row_lines_up_with_the_header(self):
        row = Tally("x", asked=10, top1=1, top5=2, form=3).row
        self.assertEqual(len(row), len(header()))


def paired_setup():
    """A dense and a sparse space over the same vocabulary."""
    stems = ["ship", "whale", "boat", "sail"]
    words, vectors, rows = [], [], []
    for i, stem in enumerate(stems):
        base = [0.0] * (len(stems) + 1)
        base[i] = 1.0
        words += [stem, stem + "s"]
        vectors.append(base)
        vectors.append([*base[:-1], 1.0])
        rows.append({i: 1.0})
        rows.append({i: 1.0, len(stems): 1.0})
    dense = Space(
        words=words,
        vectors=vectors,
        meta={"eigenvalues": [1.0] * (len(stems) + 1), "config": {}},
    )
    sparse = SparseSpace(words, SparseMatrix(rows))
    category = Category(
        rule=Rule("plural", suffix="s"),
        pairs=[(s, s + "s") for s in stems],
    )
    return dense, sparse, [category]


class TestCompare(unittest.TestCase):
    def test_both_representations_answer_the_same_questions(self):
        dense, sparse, categories = paired_setup()
        result = compare(dense, sparse, categories, limit=99)
        self.assertEqual(result.dense.asked, result.sparse.asked)
        self.assertEqual(result.dense.asked, 12)

    def test_two_perfect_representations_never_disagree(self):
        dense, sparse, categories = paired_setup()
        result = compare(dense, sparse, categories, limit=99)
        for metric in ("top1", "top5", "form"):
            self.assertEqual(result.discordant[metric], (0, 0))
            self.assertEqual(result.significance(metric)[2], 1.0)

    def test_disagreement_is_attributed_to_the_right_side(self):
        dense, sparse, categories = paired_setup()
        # Break the dense space so only the sparse one can answer.
        broken = Space(
            words=dense.words,
            vectors=[[1.0, 0.0] for _ in dense.words],
            meta={"eigenvalues": [1.0, 1.0], "config": {}},
        )
        result = compare(broken, sparse, categories, limit=99)
        sparse_only, dense_only, _ = result.significance("top1")
        self.assertGreater(sparse_only, 0)
        self.assertEqual(dense_only, 0)

    def test_words_missing_from_the_sparse_space_are_skipped(self):
        dense, sparse, categories = paired_setup()
        shrunk = SparseSpace(["ship"], SparseMatrix([{0: 1.0}]))
        result = compare(dense, shrunk, categories, limit=99)
        self.assertEqual(result.dense.asked, 0)

    def test_labels_identify_the_representations(self):
        dense, sparse, categories = paired_setup()
        result = compare(dense, sparse, categories, limit=4)
        self.assertIn("64", result.dense.label)
        self.assertIn("4000", result.sparse.label)


class TestComparison(unittest.TestCase):
    def test_significance_reports_counts_and_p(self):
        comparison = Comparison(
            dense=Tally("d"), sparse=Tally("s"),
            discordant={"form": (66, 105)},
        )
        sparse_only, dense_only, p = comparison.significance("form")
        self.assertEqual((sparse_only, dense_only), (66, 105))
        self.assertAlmostEqual(p, 0.0037, places=4)


if __name__ == "__main__":
    unittest.main()
