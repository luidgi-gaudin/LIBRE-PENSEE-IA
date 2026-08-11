from __future__ import annotations

import unittest

from sens.counts import cooccurrence


class TestCooccurrence(unittest.TestCase):
    def test_adjacent_pair_is_counted_both_ways(self):
        m = cooccurrence([0, 1], size=2, window=1)
        self.assertEqual(m.rows[0][1], 1.0)
        self.assertEqual(m.rows[1][0], 1.0)

    def test_matrix_is_symmetric(self):
        m = cooccurrence([0, 1, 2, 1, 0, 2], size=3, window=2)
        for i, row in enumerate(m.rows):
            for j, v in row.items():
                self.assertAlmostEqual(v, m.rows[j].get(i, 0.0))

    def test_window_bounds_the_reach(self):
        # With window 1, the ends of a three-token sentence never meet.
        m = cooccurrence([0, 1, 2], size=3, window=1)
        self.assertNotIn(2, m.rows[0])

    def test_wider_window_connects_them(self):
        m = cooccurrence([0, 1, 2], size=3, window=2)
        self.assertIn(2, m.rows[0])

    def test_harmonic_weighting_decays_with_distance(self):
        m = cooccurrence([0, 1, 2], size=3, window=2, harmonic=True)
        self.assertAlmostEqual(m.rows[0][1], 1.0)      # distance 1
        self.assertAlmostEqual(m.rows[0][2], 0.5)      # distance 2

    def test_flat_weighting_treats_all_distances_alike(self):
        m = cooccurrence([0, 1, 2], size=3, window=2, harmonic=False)
        self.assertAlmostEqual(m.rows[0][1], 1.0)
        self.assertAlmostEqual(m.rows[0][2], 1.0)

    def test_repeated_pairs_accumulate(self):
        m = cooccurrence([0, 1, 0, 1], size=2, window=1)
        self.assertAlmostEqual(m.rows[0][1], 3.0)

    def test_a_word_next_to_itself_counts_on_its_diagonal(self):
        m = cooccurrence([0, 0], size=1, window=1)
        # Both directions of the same pair land in the same cell.
        self.assertAlmostEqual(m.rows[0][0], 2.0)

    def test_min_weight_prunes_weak_pairs(self):
        # `0` and `2` meet once at distance 2, contributing 0.5.
        m = cooccurrence([0, 1, 2], size=3, window=2, min_weight=1.0)
        self.assertNotIn(2, m.rows[0])
        self.assertIn(1, m.rows[0])

    def test_pruning_stays_symmetric(self):
        ids = [0, 1, 2, 3, 0, 1, 0, 1, 2]
        m = cooccurrence(ids, size=4, window=3, min_weight=0.75)
        for i, row in enumerate(m.rows):
            for j in row:
                self.assertIn(i, m.rows[j])

    def test_empty_input_gives_an_empty_matrix(self):
        m = cooccurrence([], size=3, window=2)
        self.assertEqual(m.nnz, 0)
        self.assertEqual(m.n, 3)

    def test_total_weight_matches_the_number_of_pairs_seen(self):
        # Flat weighting over 4 tokens with window 1 gives 3 adjacent pairs,
        # each recorded twice for symmetry.
        m = cooccurrence([0, 1, 2, 3], size=4, window=1, harmonic=False)
        total = sum(v for row in m.rows for v in row.values())
        self.assertAlmostEqual(total, 6.0)


if __name__ == "__main__":
    unittest.main()
