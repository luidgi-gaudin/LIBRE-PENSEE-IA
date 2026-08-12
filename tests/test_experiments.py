"""Tests for reweighting, ablation, and the experiments built on them.

An experiment that silently compares two different things will produce a
confident number and a wrong conclusion. Most of what is checked here is
that the comparisons stay fair: that a reweighting really is the same
factorisation, and that a dimension-matched ablation really is matched.
"""

from __future__ import annotations

import unittest

from sens.experiments import (
    DEFAULT_POWERS,
    Result,
    ablate_signs,
    header,
    sweep_power,
)
from sens.evaluate import Score
from sens.space import Space


def factorised(power: float = 0.5) -> Space:
    """A space carrying the eigenvalue metadata a rebuild would leave.

    Vectors are stored already scaled by `|eigenvalue| ** power`, exactly as
    the pipeline writes them, so `rescaled` has something real to undo.
    """
    values = [4.0, -2.0, 1.0]
    raw = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    scale = [abs(v) ** power for v in values]
    return Space(
        words=["alpha", "beta", "gamma", "delta"],
        vectors=[[x * s for x, s in zip(row, scale)] for row in raw],
        meta={"eigenvalues": values, "config": {"eigenvalue_power": power}},
    )


def morphological() -> Space:
    """A space with enough regular plurals to generate a real benchmark."""
    stems = ["ship", "whale", "boat", "sail", "rope", "mast"]
    words, vectors = [], []
    for i, stem in enumerate(stems):
        base = [0.0] * (len(stems) + 1)
        base[i] = 1.0
        words += [stem, stem + "s"]
        vectors.append(base)
        vectors.append([*base[:-1], 1.0])
    return Space(
        words=words,
        vectors=vectors,
        meta={
            "eigenvalues": [5.0, 4.0, -3.0, 2.0, 1.5, -1.0, 0.5],
            "config": {"eigenvalue_power": 0.5},
        },
    )


class TestRescaled(unittest.TestCase):
    def test_rescaling_to_the_current_power_changes_nothing(self):
        space = factorised(0.5)
        again = space.rescaled(0.5)
        for a_row, b_row in zip(space.vectors, again.vectors):
            for a, b in zip(a_row, b_row):
                self.assertAlmostEqual(a, b, places=12)

    def test_rescaling_to_zero_strips_the_eigenvalue_weighting(self):
        # At power 0 every retained direction should count the same, so the
        # unit basis vectors must come back exactly as they went in.
        space = factorised(0.5).rescaled(0.0)
        self.assertAlmostEqual(space.vectors[0][0], 1.0, places=12)
        self.assertAlmostEqual(space.vectors[1][1], 1.0, places=12)
        self.assertAlmostEqual(space.vectors[3][1], 1.0, places=12)

    def test_rescaling_uses_absolute_eigenvalues(self):
        # The second eigenvalue is -2. Its scale factor at power 1 must be
        # 2, not -2, or the axis would flip and the geometry with it.
        space = factorised(0.0).rescaled(1.0)
        self.assertAlmostEqual(space.vectors[1][1], 2.0, places=12)

    def test_round_trip_returns_the_original(self):
        original = factorised(0.5)
        there_and_back = original.rescaled(1.0).rescaled(0.5)
        for a_row, b_row in zip(original.vectors, there_and_back.vectors):
            for a, b in zip(a_row, b_row):
                self.assertAlmostEqual(a, b, places=10)

    def test_the_new_power_is_recorded(self):
        space = factorised(0.5).rescaled(0.75)
        self.assertEqual(space.meta["config"]["eigenvalue_power"], 0.75)

    def test_words_and_eigenvalues_survive(self):
        space = factorised().rescaled(1.0)
        self.assertEqual(space.words, ["alpha", "beta", "gamma", "delta"])
        self.assertEqual(space.meta["eigenvalues"], [4.0, -2.0, 1.0])

    def test_it_actually_changes_the_geometry(self):
        # If reweighting were a no-op on cosines there would be nothing to
        # sweep, so this guards the premise of the whole experiment.
        before = factorised(0.5).similarity("alpha", "delta")
        after = factorised(0.5).rescaled(0.0).similarity("alpha", "delta")
        self.assertNotAlmostEqual(before, after, places=4)

    def test_a_space_without_eigenvalues_refuses(self):
        space = Space(words=["a", "b"], vectors=[[1.0], [2.0]])
        with self.assertRaises(ValueError):
            space.rescaled(1.0)

    def test_mismatched_eigenvalue_count_refuses(self):
        space = Space(
            words=["a"], vectors=[[1.0, 2.0]], meta={"eigenvalues": [1.0]}
        )
        with self.assertRaises(ValueError):
            space.rescaled(1.0)


class TestSubspace(unittest.TestCase):
    def test_keeps_only_the_listed_dimensions(self):
        space = factorised().subspace([0, 2])
        self.assertEqual(space.dim, 2)
        self.assertEqual(space.vectors[3], [
            factorised().vectors[3][0], factorised().vectors[3][2]
        ])

    def test_eigenvalues_are_sliced_to_match(self):
        space = factorised().subspace([0, 2])
        self.assertEqual(space.meta["eigenvalues"], [4.0, 1.0])

    def test_words_are_preserved(self):
        self.assertEqual(len(factorised().subspace([1])), 4)

    def test_order_of_the_kept_list_is_respected(self):
        space = factorised().subspace([2, 0])
        self.assertEqual(space.meta["eigenvalues"], [1.0, 4.0])

    def test_a_subspace_can_be_rescaled(self):
        space = factorised(0.5).subspace([0, 1]).rescaled(0.0)
        self.assertAlmostEqual(space.vectors[0][0], 1.0, places=12)


class TestSignPartitions(unittest.TestCase):
    def test_positive_and_negative_split_the_dimensions(self):
        space = factorised()
        self.assertEqual(space.positive_dimensions(), [0, 2])
        self.assertEqual(space.negative_dimensions(), [1])

    def test_they_are_disjoint_and_complete(self):
        space = morphological()
        positive = set(space.positive_dimensions())
        negative = set(space.negative_dimensions())
        self.assertEqual(positive & negative, set())
        self.assertEqual(positive | negative, set(range(space.dim)))


class TestSweepPower(unittest.TestCase):
    def test_one_result_per_power(self):
        results = sweep_power(morphological(), powers=(0.0, 0.5, 1.0), limit=6)
        self.assertEqual([r.label for r in results],
                         ["power 0.00", "power 0.50", "power 1.00"])

    def test_dimension_count_is_unchanged_by_reweighting(self):
        results = sweep_power(morphological(), limit=6)
        self.assertTrue(all(r.dims == 7 for r in results))

    def test_default_powers_span_zero_to_one(self):
        self.assertEqual(DEFAULT_POWERS[0], 0.0)
        self.assertEqual(DEFAULT_POWERS[-1], 1.0)

    def test_results_are_reproducible(self):
        space = morphological()
        first = sweep_power(space, powers=(0.5,), limit=6, seed=3)
        second = sweep_power(space, powers=(0.5,), limit=6, seed=3)
        self.assertEqual(first[0].score.top1, second[0].score.top1)


class TestAblateSigns(unittest.TestCase):
    def test_the_first_two_rows_are_dimension_matched(self):
        # This is the whole validity of the experiment: comparing a 49-dim
        # space against a 64-dim one would measure dimensionality, not sign.
        results = ablate_signs(morphological(), limit=6)
        self.assertEqual(results[0].dims, results[1].dims)

    def test_it_reports_four_configurations(self):
        results = ablate_signs(morphological(), limit=6)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[-1].dims, 7)

    def test_a_space_with_no_negative_eigenvalues_has_nothing_to_ablate(self):
        space = Space(
            words=["a", "b"],
            vectors=[[1.0, 0.0], [0.0, 1.0]],
            meta={"eigenvalues": [2.0, 1.0], "config": {}},
        )
        self.assertEqual(ablate_signs(space, limit=2), [])


class TestFormatting(unittest.TestCase):
    def test_header_names_every_column(self):
        for column in ("configuration", "dims", "top1", "top5", "form"):
            self.assertIn(column, header())

    def test_row_renders_percentages(self):
        row = Result("power 0.50", 64, Score("m", "ALL", 100, 4, 11, 18)).row
        self.assertIn("power 0.50", row)
        self.assertIn("4.0%", row)
        self.assertIn("64", row)

    def test_row_lines_up_with_the_header(self):
        row = Result("x", 1, Score("m", "ALL", 10, 1, 1, 1)).row
        self.assertEqual(len(row), len(header()))


if __name__ == "__main__":
    unittest.main()


class TestDimensionCurve(unittest.TestCase):
    """The curve reuses one factorisation, so the prefix property matters."""

    def setUp(self):
        from sens.heldout import Prepared
        from sens.linalg import SparseMatrix

        space = morphological()
        # A truth table where the two halves of the vocabulary share context.
        rows = []
        for i in range(len(space.words)):
            rows.append({0: 1.0, 1: float(i % 3) + 0.5, 2: float(i % 2)})
        self.prepared = Prepared(
            space=space,
            truth=SparseMatrix(rows),
            vocab=None,
            pairs=[(0, 2), (1, 3), (0, 4), (2, 5)],
            train_tokens=100,
            test_tokens=100,
        )

    def test_one_row_per_dimension(self):
        from sens.experiments import dimension_curve

        rows = dimension_curve(self.prepared, dims=(2, 4, 7), limit=6)
        self.assertEqual([r.dims for r in rows], [2, 4, 7])

    def test_each_row_reports_both_metrics(self):
        from sens.experiments import dimension_curve

        for row in dimension_curve(self.prepared, dims=(4,), limit=6):
            self.assertIsInstance(row.heldout, float)
            self.assertIsInstance(row.form, float)
            self.assertGreaterEqual(row.form, 0.0)
            self.assertLessEqual(row.form, 1.0)

    def test_smaller_dimensions_are_prefixes_of_larger_ones(self):
        # The whole curve costs one build only because this holds: the
        # factorisation returns directions ordered by |eigenvalue|, so
        # taking the first k is taking the k strongest.
        full = self.prepared.space
        prefix = full.subspace(list(range(3)))
        for row_full, row_prefix in zip(full.vectors, prefix.vectors):
            self.assertEqual(row_full[:3], row_prefix)

    def test_row_lines_up_with_the_header(self):
        from sens.experiments import DimensionRow, dimension_header

        row = DimensionRow(dims=64, heldout=0.6, top5=0.1, form=0.2).row
        self.assertEqual(len(row), len(dimension_header()))

    def test_header_names_both_metrics(self):
        from sens.experiments import dimension_header

        for column in ("dims", "held-out", "top5", "form"):
            self.assertIn(column, dimension_header())
