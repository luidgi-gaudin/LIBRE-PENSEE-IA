from __future__ import annotations

import math
import unittest

from sens.linalg import SparseMatrix
from sens.weight import WEIGHTINGS, log_counts, pmi, ppmi, raw


def sparse(rows) -> SparseMatrix:
    return SparseMatrix([{j: v for j, v in enumerate(r) if v} for r in rows])


class TestPPMI(unittest.TestCase):
    def test_matches_a_hand_computed_value(self):
        # A 2x2 table where every quantity is easy to write down.
        #   C = [[4, 1],
        #        [1, 4]]
        # marginals are 5 and 5, total 10.
        # With alpha = 1 the smoothed context distribution is just p(j) = 0.5.
        #   pmi(0,0) = log( (4/10) / (0.5 * 0.5) ) = log(1.6)
        m = ppmi(sparse([[4.0, 1.0], [1.0, 4.0]]), alpha=1.0)
        self.assertAlmostEqual(m.rows[0][0], math.log(1.6), places=12)

    def test_negative_scores_are_clipped_away(self):
        # pmi(0,1) = log( (1/10) / (0.5 * 0.5) ) = log(0.4) < 0
        m = ppmi(sparse([[4.0, 1.0], [1.0, 4.0]]), alpha=1.0)
        self.assertNotIn(1, m.rows[0])

    def test_independent_table_scores_zero_everywhere(self):
        # An outer product of marginals is exactly the independence model,
        # so every PMI is log(1) = 0. In floating point a handful land a few
        # ulps above zero and slip past the clip, which is why this asserts
        # negligible rather than absent.
        table = [[1.0, 2.0, 3.0],
                 [2.0, 4.0, 6.0],
                 [3.0, 6.0, 9.0]]
        m = ppmi(sparse(table), alpha=1.0)
        residual = [v for row in m.rows for v in row.values()]
        self.assertTrue(all(v < 1e-12 for v in residual), residual)

    def test_a_dedicated_pair_scores_high(self):
        # Two words that only ever appear with each other, in a big corpus
        # of other traffic, should get a large positive score.
        table = [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 1000.0]]
        m = ppmi(sparse(table), alpha=1.0)
        self.assertGreater(m.rows[0][1], 4.0)

    def test_context_smoothing_lowers_rare_context_scores(self):
        # Column 2 is a rare context. Flattening the context distribution
        # (alpha < 1) must reduce the bonus it gets for being rare. Here it
        # reduces it past zero and the entry disappears entirely, which is
        # the intended effect at full strength.
        table = [[10.0, 10.0, 1.0], [10.0, 10.0, 0.0], [1.0, 0.0, 0.0]]
        plain = ppmi(sparse(table), alpha=1.0)
        smoothed = ppmi(sparse(table), alpha=0.75)
        self.assertGreater(plain.rows[0][2], 0.0)
        self.assertLess(smoothed.rows[0].get(2, 0.0), plain.rows[0][2])

    def test_context_smoothing_leaves_common_contexts_alone(self):
        # The flattening is a redistribution, so what the rare context loses
        # the common ones must gain.
        table = [[10.0, 10.0, 1.0], [10.0, 10.0, 0.0], [1.0, 0.0, 0.0]]
        plain = ppmi(sparse(table), alpha=1.0)
        smoothed = ppmi(sparse(table), alpha=0.75)
        self.assertGreater(smoothed.rows[0][1], plain.rows[0][1])

    def test_shift_removes_weak_evidence(self):
        table = [[4.0, 1.0], [1.0, 4.0]]
        plain = ppmi(sparse(table), alpha=1.0)
        shifted = ppmi(sparse(table), alpha=1.0, shift=1.5)
        self.assertLess(shifted.nnz, plain.nnz + 1)
        self.assertAlmostEqual(
            shifted.rows[0][0], plain.rows[0][0] - math.log(1.5), places=12
        )

    def test_result_is_symmetric_for_a_symmetric_input(self):
        table = [[3.0, 1.0, 0.0], [1.0, 4.0, 2.0], [0.0, 2.0, 5.0]]
        m = ppmi(sparse(table))
        for i, row in enumerate(m.rows):
            for j, v in row.items():
                self.assertAlmostEqual(v, m.rows[j][i], places=12)

    def test_shape_is_preserved(self):
        m = ppmi(sparse([[1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(m.n, 2)

    def test_empty_matrix_is_handled(self):
        m = ppmi(SparseMatrix([{}, {}, {}]))
        self.assertEqual(m.n, 3)
        self.assertEqual(m.nnz, 0)

    def test_a_word_with_no_counts_gets_an_empty_row(self):
        m = ppmi(sparse([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))
        self.assertEqual(m.rows[2], {})

    def test_every_surviving_score_is_positive(self):
        table = [[7.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 9.0]]
        m = ppmi(sparse(table))
        self.assertTrue(all(v > 0.0 for row in m.rows for v in row.values()))


class TestRaw(unittest.TestCase):
    def test_it_returns_the_counts_unchanged(self):
        table = [[3.0, 1.0], [1.0, 4.0]]
        self.assertEqual(raw(sparse(table)).rows[0], {0: 3.0, 1: 1.0})

    def test_it_copies_rather_than_aliasing(self):
        counts = sparse([[1.0, 2.0], [2.0, 1.0]])
        result = raw(counts)
        result.rows[0][0] = 99.0
        self.assertEqual(counts.rows[0][0], 1.0)

    def test_shape_is_preserved(self):
        self.assertEqual(raw(SparseMatrix([{}, {}, {}])).n, 3)


class TestLogCounts(unittest.TestCase):
    def test_it_is_log_one_plus_count(self):
        result = log_counts(sparse([[3.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(result.rows[0][0], math.log(4.0), places=12)
        self.assertAlmostEqual(result.rows[1][1], math.log(2.0), places=12)

    def test_it_is_monotone_in_the_count(self):
        result = log_counts(sparse([[1.0, 5.0, 50.0]]))
        row = result.rows[0]
        self.assertLess(row[0], row[1])
        self.assertLess(row[1], row[2])

    def test_it_compresses_the_range(self):
        # The whole point: a hundredfold difference in counts becomes a
        # fivefold difference in weight.
        result = log_counts(sparse([[1.0, 100.0]])).rows[0]
        self.assertLess(result[1] / result[0], 10.0)

    def test_zero_counts_are_dropped(self):
        self.assertNotIn(1, log_counts(SparseMatrix([{0: 2.0, 1: 0.0}])).rows[0])


class TestUnclippedPMI(unittest.TestCase):
    def test_it_keeps_negative_scores(self):
        # pmi(0,1) for this table is log(0.4), which ppmi discards.
        table = [[4.0, 1.0], [1.0, 4.0]]
        self.assertAlmostEqual(
            pmi(sparse(table), alpha=1.0).rows[0][1], math.log(0.4), places=12
        )

    def test_ppmi_is_this_with_the_negatives_removed(self):
        table = [[7.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 9.0]]
        signed = pmi(sparse(table))
        clipped = ppmi(sparse(table))
        expected = {
            i: {j: v for j, v in row.items() if v > 0.0}
            for i, row in enumerate(signed.rows)
        }
        for i, row in enumerate(clipped.rows):
            self.assertEqual(row, expected[i])

    def test_it_keeps_more_entries_than_ppmi(self):
        table = [[4.0, 1.0], [1.0, 4.0]]
        self.assertGreater(pmi(sparse(table)).nnz, ppmi(sparse(table)).nnz)

    def test_it_is_symmetric_for_a_symmetric_input(self):
        table = [[3.0, 1.0, 0.0], [1.0, 4.0, 2.0], [0.0, 2.0, 5.0]]
        result = pmi(sparse(table))
        for i, row in enumerate(result.rows):
            for j, v in row.items():
                self.assertAlmostEqual(v, result.rows[j][i], places=12)


class TestRegistry(unittest.TestCase):
    def test_every_name_maps_to_its_function(self):
        self.assertIs(WEIGHTINGS["ppmi"], ppmi)
        self.assertIs(WEIGHTINGS["pmi"], pmi)
        self.assertIs(WEIGHTINGS["log"], log_counts)
        self.assertIs(WEIGHTINGS["raw"], raw)

    def test_every_weighting_accepts_the_same_call(self):
        # The pipeline passes alpha and shift to all of them, so the two
        # that ignore those still have to tolerate being handed them.
        table = sparse([[4.0, 1.0], [1.0, 4.0]])
        for name, weigh in WEIGHTINGS.items():
            result = weigh(table, alpha=1.0, shift=1.0)
            self.assertEqual(result.n, 2, name)

    def test_the_pipeline_default_is_in_the_registry(self):
        from sens.pipeline import Config

        self.assertIn(Config().weighting, WEIGHTINGS)


if __name__ == "__main__":
    unittest.main()
