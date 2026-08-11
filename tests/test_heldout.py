"""Tests for the held-out generalisation measure.

The statistics get the most attention here. A correlation coefficient will
return a confident number for any two lists of floats, including lists that
mean nothing, so the only defence is to check it against cases whose answer
is known in advance.
"""

from __future__ import annotations

import unittest

from sens.heldout import (
    Prepared,
    correlation,
    pearson,
    sparse_cosine,
    spearman,
    split_blocks,
    sweep_power,
    _ranks,
)
from sens.linalg import SparseMatrix
from sens.space import Space


class TestSplitBlocks(unittest.TestCase):
    def test_alternates_whole_blocks(self):
        tokens = [str(i) for i in range(10)]
        train, test = split_blocks(tokens, block=2)
        self.assertEqual(train, ["0", "1", "4", "5", "8", "9"])
        self.assertEqual(test, ["2", "3", "6", "7"])

    def test_halves_are_disjoint_and_complete(self):
        tokens = [str(i) for i in range(97)]
        train, test = split_blocks(tokens, block=7)
        self.assertEqual(len(train) + len(test), 97)
        self.assertEqual(set(train) & set(test), set())

    def test_a_short_corpus_all_lands_in_train(self):
        train, test = split_blocks(["a", "b"], block=10)
        self.assertEqual(train, ["a", "b"])
        self.assertEqual(test, [])

    def test_empty_input(self):
        self.assertEqual(split_blocks([], block=4), ([], []))

    def test_the_halves_stay_roughly_balanced(self):
        tokens = [str(i) for i in range(1000)]
        train, test = split_blocks(tokens, block=50)
        self.assertEqual(len(train), len(test))


class TestSparseCosine(unittest.TestCase):
    def test_identical_rows_score_one(self):
        row = {1: 3.0, 5: 4.0}
        self.assertAlmostEqual(sparse_cosine(row, dict(row)), 1.0)

    def test_disjoint_rows_score_zero(self):
        self.assertEqual(sparse_cosine({1: 1.0}, {2: 1.0}), 0.0)

    def test_empty_rows_score_zero(self):
        self.assertEqual(sparse_cosine({}, {1: 1.0}), 0.0)
        self.assertEqual(sparse_cosine({1: 1.0}, {}), 0.0)

    def test_matches_a_hand_computed_value(self):
        # (3,4) . (4,3) = 24, norms 5 and 5, so 24/25.
        a = {0: 3.0, 1: 4.0}
        b = {0: 4.0, 1: 3.0}
        self.assertAlmostEqual(sparse_cosine(a, b), 0.96)

    def test_scale_does_not_matter(self):
        a = {0: 1.0, 1: 2.0}
        b = {0: 10.0, 1: 20.0}
        self.assertAlmostEqual(sparse_cosine(a, b), 1.0)

    def test_is_symmetric(self):
        a = {0: 1.0, 3: 2.5}
        b = {0: 0.5, 3: 1.0, 9: 4.0}
        self.assertAlmostEqual(sparse_cosine(a, b), sparse_cosine(b, a))

    def test_argument_order_does_not_change_the_result(self):
        # The implementation swaps so it iterates the shorter row; that must
        # be invisible from outside.
        small = {0: 1.0}
        large = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        self.assertAlmostEqual(
            sparse_cosine(small, large), sparse_cosine(large, small)
        )


class TestRanks(unittest.TestCase):
    def test_simple_ordering(self):
        self.assertEqual(_ranks([10.0, 30.0, 20.0]), [1.0, 3.0, 2.0])

    def test_ties_share_the_average_rank(self):
        # Two values tied for ranks 2 and 3 both get 2.5.
        self.assertEqual(_ranks([1.0, 5.0, 5.0, 9.0]), [1.0, 2.5, 2.5, 4.0])

    def test_all_equal_values_all_share_one_rank(self):
        self.assertEqual(_ranks([7.0, 7.0, 7.0]), [2.0, 2.0, 2.0])

    def test_single_value(self):
        self.assertEqual(_ranks([4.0]), [1.0])


class TestPearson(unittest.TestCase):
    def test_perfect_positive(self):
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)

    def test_perfect_negative(self):
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0)

    def test_a_constant_series_has_no_correlation(self):
        self.assertEqual(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), 0.0)

    def test_too_few_points(self):
        self.assertEqual(pearson([1.0], [2.0]), 0.0)


class TestSpearman(unittest.TestCase):
    def test_monotone_but_nonlinear_still_scores_one(self):
        # This is the reason for using rank rather than value: the space is
        # only ever asked to order words, never to reproduce magnitudes.
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 8.0, 27.0, 64.0]
        self.assertAlmostEqual(spearman(xs, ys), 1.0)
        self.assertLess(pearson(xs, ys), 1.0)

    def test_reversed_order_scores_minus_one(self):
        self.assertAlmostEqual(
            spearman([1.0, 2.0, 3.0], [9.0, 5.0, 1.0]), -1.0
        )

    def test_handles_ties(self):
        value = spearman([1.0, 1.0, 2.0], [3.0, 3.0, 9.0])
        self.assertAlmostEqual(value, 1.0)


def toy_prepared() -> Prepared:
    """A space and a truth table that agree perfectly on two pairs.

    `alpha` and `beta` share a context in the held-out table and sit on top
    of each other in the space; `gamma` shares nothing and points elsewhere.
    A correct correlation must come out positive.
    """
    space = Space(
        words=["alpha", "beta", "gamma"],
        vectors=[[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]],
        meta={"eigenvalues": [2.0, 1.0], "config": {"eigenvalue_power": 0.5}},
    )
    truth = SparseMatrix([
        {0: 1.0, 1: 1.0},
        {0: 1.0, 1: 1.0},
        {2: 1.0},
    ])
    return Prepared(
        space=space,
        truth=truth,
        vocab=None,
        pairs=[(0, 1), (0, 2), (1, 2)],
        train_tokens=10,
        test_tokens=10,
    )


class TestCorrelation(unittest.TestCase):
    def test_skips_pairs_with_no_held_out_evidence(self):
        prepared = toy_prepared()
        _, used = correlation(prepared.space, prepared)
        # `gamma` shares no context with the others, so two of the three
        # pairs have a zero truth value and cannot be scored.
        self.assertEqual(used, 1)

    def test_agreement_gives_a_positive_correlation(self):
        space = Space(
            words=["a", "b", "c", "d"],
            vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            meta={"eigenvalues": [1.0, 1.0], "config": {}},
        )
        # Every row shares a little context with every other, so that all
        # four pairs are scorable. Without the small cross terms the two
        # groups would be orthogonal, their truth cosine would be exactly
        # zero, and the pairs would be skipped as unmeasurable.
        truth = SparseMatrix([
            {0: 1.0, 1: 1.0, 2: 0.1},
            {0: 1.0, 1: 0.9, 2: 0.1},
            {2: 1.0, 3: 1.0, 0: 0.1},
            {2: 0.9, 3: 1.0, 0: 0.1},
        ])
        prepared = Prepared(
            space=space, truth=truth, vocab=None,
            pairs=[(0, 1), (2, 3), (0, 2), (1, 3)],
            train_tokens=1, test_tokens=1,
        )
        rho, used = correlation(space, prepared)
        self.assertEqual(used, 4)
        self.assertGreater(rho, 0.0)

    def test_no_scorable_pairs_returns_zero(self):
        space = Space(words=["a", "b"], vectors=[[1.0], [1.0]],
                      meta={"eigenvalues": [1.0], "config": {}})
        prepared = Prepared(
            space=space, truth=SparseMatrix([{}, {}]), vocab=None,
            pairs=[(0, 1)], train_tokens=1, test_tokens=1,
        )
        self.assertEqual(correlation(space, prepared), (0.0, 0))


class TestSweepPower(unittest.TestCase):
    def test_returns_one_row_per_power(self):
        results = sweep_power(toy_prepared(), (0.0, 0.5, 1.0))
        self.assertEqual([r[0] for r in results], [0.0, 0.5, 1.0])

    def test_each_row_reports_the_pair_count(self):
        for _, _, used in sweep_power(toy_prepared(), (0.5,)):
            self.assertEqual(used, 1)


class TestPrepared(unittest.TestCase):
    def test_coverage_is_pairs_over_vocabulary(self):
        prepared = toy_prepared()
        prepared.vocab = ["a", "b", "c"]
        self.assertAlmostEqual(prepared.coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
