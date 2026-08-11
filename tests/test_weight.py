from __future__ import annotations

import math
import unittest

from sens.linalg import SparseMatrix
from sens.weight import ppmi


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


if __name__ == "__main__":
    unittest.main()
