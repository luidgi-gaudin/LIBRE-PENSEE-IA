"""The numerics are the part that can be silently wrong.

Nearest-neighbour output looks convincing even when the factorisation is
broken, because words that co-occur will cluster under almost any transform.
So these tests check the linear algebra against cases with known answers
rather than against anything that requires judgement.
"""

from __future__ import annotations

import math
import random
import unittest

from sens.linalg import (
    SparseMatrix,
    block_krylov_eigh,
    apply_right,
    cholesky,
    cross,
    gram,
    invert_lower,
    jacobi_eigh,
    orthonormalize,
    principal_angles,
    randomized_eigh,
    unit,
)


def dense_to_sparse(dense) -> SparseMatrix:
    return SparseMatrix(
        [{j: v for j, v in enumerate(row) if v != 0.0} for row in dense]
    )


def random_symmetric(n: int, rng: random.Random):
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = rng.gauss(0.0, 1.0)
            m[i][j] = v
            m[j][i] = v
    return m


def matmul(a, b):
    return [
        [sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def transpose(m):
    return [list(col) for col in zip(*m)]


class TestSparse(unittest.TestCase):
    def test_dot_dense_matches_naive_product(self):
        rng = random.Random(1)
        dense = [[rng.choice([0.0, 0.0, rng.random()]) for _ in range(6)]
                 for _ in range(6)]
        x = [[rng.random() for _ in range(3)] for _ in range(6)]
        expected = matmul(dense, x)
        got = dense_to_sparse(dense).dot_dense(x)
        for r_e, r_g in zip(expected, got):
            for e, g in zip(r_e, r_g):
                self.assertAlmostEqual(e, g, places=10)

    def test_nnz_counts_stored_entries(self):
        m = SparseMatrix([{0: 1.0, 2: 3.0}, {}, {1: 1.0}])
        self.assertEqual(m.nnz, 3)

    def test_to_dense_round_trips(self):
        dense = [[1.0, 0.0], [0.0, 2.0]]
        self.assertEqual(dense_to_sparse(dense).to_dense(), dense)


class TestDenseHelpers(unittest.TestCase):
    def test_gram_equals_transpose_times_self(self):
        rng = random.Random(2)
        x = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(9)]
        expected = matmul(transpose(x), x)
        for r_e, r_g in zip(expected, gram(x)):
            for e, g in zip(r_e, r_g):
                self.assertAlmostEqual(e, g, places=10)

    def test_cross_equals_a_transpose_times_b(self):
        rng = random.Random(3)
        a = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(7)]
        b = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(7)]
        expected = matmul(transpose(a), b)
        for r_e, r_g in zip(expected, cross(a, b)):
            for e, g in zip(r_e, r_g):
                self.assertAlmostEqual(e, g, places=10)

    def test_apply_right_equals_product(self):
        rng = random.Random(4)
        x = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(6)]
        m = [[rng.gauss(0, 1) for _ in range(2)] for _ in range(4)]
        expected = matmul(x, m)
        for r_e, r_g in zip(expected, apply_right(x, m)):
            for e, g in zip(r_e, r_g):
                self.assertAlmostEqual(e, g, places=10)


class TestCholesky(unittest.TestCase):
    def test_factor_reconstructs_the_matrix(self):
        rng = random.Random(5)
        b = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(10)]
        a = gram(b)  # positive definite by construction
        low = cholesky(a)
        recon = matmul(low, transpose(low))
        for r_a, r_r in zip(a, recon):
            for x, y in zip(r_a, r_r):
                self.assertAlmostEqual(x, y, places=8)

    def test_factor_is_lower_triangular(self):
        a = gram([[1.0, 2.0], [3.0, 1.0], [0.5, 0.25]])
        low = cholesky(a)
        self.assertAlmostEqual(low[0][1], 0.0)

    def test_ridge_keeps_singular_input_from_raising(self):
        # Two identical columns: the Gram matrix is exactly rank one.
        a = gram([[1.0, 1.0], [2.0, 2.0]])
        low = cholesky(a)
        self.assertTrue(all(math.isfinite(v) for row in low for v in row))

    def test_inverse_is_an_inverse(self):
        a = gram([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]])
        low = cholesky(a)
        identity = matmul(low, invert_lower(low))
        for i, row in enumerate(identity):
            for j, v in enumerate(row):
                self.assertAlmostEqual(v, 1.0 if i == j else 0.0, places=8)


class TestOrthonormalize(unittest.TestCase):
    def test_columns_become_orthonormal(self):
        rng = random.Random(6)
        x = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(40)]
        q = orthonormalize(x)
        g = gram(q)
        for i, row in enumerate(g):
            for j, v in enumerate(row):
                self.assertAlmostEqual(v, 1.0 if i == j else 0.0, places=8)

    def test_survives_a_nearly_dependent_column(self):
        rng = random.Random(7)
        x = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(20)]
        for row in x:
            row.append(row[0] + 1e-9 * rng.gauss(0, 1))
        q = orthonormalize(x)
        self.assertTrue(all(math.isfinite(v) for row in q for v in row))
        g = gram(q)
        # The three genuine directions must still come out orthonormal even
        # though the fourth carries no information.
        for i in range(3):
            self.assertAlmostEqual(g[i][i], 1.0, places=6)
            for j in range(3):
                if i != j:
                    self.assertAlmostEqual(g[i][j], 0.0, places=6)

    def test_span_is_preserved(self):
        rng = random.Random(8)
        x = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(12)]
        q = orthonormalize(x)
        # Every original column must be reproducible from Q, i.e. projecting
        # it onto Q and back must return it unchanged.
        coeffs = cross(q, x)
        recon = apply_right(q, coeffs)
        for r_x, r_r in zip(x, recon):
            for a, b in zip(r_x, r_r):
                self.assertAlmostEqual(a, b, places=7)


class TestJacobi(unittest.TestCase):
    def test_diagonal_matrix_is_returned_sorted(self):
        values, _ = jacobi_eigh([[1.0, 0.0, 0.0],
                                 [0.0, 5.0, 0.0],
                                 [0.0, 0.0, 3.0]])
        self.assertAlmostEqual(values[0], 5.0)
        self.assertAlmostEqual(values[1], 3.0)
        self.assertAlmostEqual(values[2], 1.0)

    def test_known_two_by_two(self):
        # [[2, 1], [1, 2]] has eigenvalues 3 and 1.
        values, vectors = jacobi_eigh([[2.0, 1.0], [1.0, 2.0]])
        self.assertAlmostEqual(values[0], 3.0, places=10)
        self.assertAlmostEqual(values[1], 1.0, places=10)
        leading = [row[0] for row in vectors]
        self.assertAlmostEqual(abs(leading[0]), abs(leading[1]), places=10)

    def test_reconstructs_a_random_symmetric_matrix(self):
        rng = random.Random(9)
        a = random_symmetric(8, rng)
        values, vectors = jacobi_eigh(a)
        diag = [[values[i] if i == j else 0.0 for j in range(8)]
                for i in range(8)]
        recon = matmul(matmul(vectors, diag), transpose(vectors))
        for r_a, r_r in zip(a, recon):
            for x, y in zip(r_a, r_r):
                self.assertAlmostEqual(x, y, places=8)

    def test_eigenvectors_are_orthonormal(self):
        rng = random.Random(10)
        a = random_symmetric(6, rng)
        _, vectors = jacobi_eigh(a)
        g = gram(vectors)
        for i, row in enumerate(g):
            for j, v in enumerate(row):
                self.assertAlmostEqual(v, 1.0 if i == j else 0.0, places=8)

    def test_eigenpairs_satisfy_the_defining_equation(self):
        rng = random.Random(11)
        a = random_symmetric(5, rng)
        values, vectors = jacobi_eigh(a)
        for k, value in enumerate(values):
            v = [row[k] for row in vectors]
            av = [sum(a[i][j] * v[j] for j in range(5)) for i in range(5)]
            for x, y in zip(av, [value * c for c in v]):
                self.assertAlmostEqual(x, y, places=8)


class TestRandomizedEigh(unittest.TestCase):
    def test_recovers_a_planted_spectrum(self):
        # Build a matrix with eigenvalues we choose, in a rotated basis, then
        # check the approximation finds the large ones.
        rng = random.Random(12)
        n = 30
        basis = orthonormalize(
            [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        )
        planted = [20.0, 12.0, 7.0, 3.0] + [0.2] * (n - 4)
        diag = [[planted[i] if i == j else 0.0 for j in range(n)]
                for i in range(n)]
        dense = matmul(matmul(basis, diag), transpose(basis))

        values, _ = randomized_eigh(
            dense_to_sparse(dense), k=4, oversample=12,
            power_iterations=4, rng=random.Random(13),
        )
        for expected, got in zip(planted[:4], values):
            self.assertAlmostEqual(expected, got, places=5)

    def test_finds_large_negative_eigenvalues_too(self):
        # PPMI matrices are symmetric but indefinite. A truncated SVD keeps
        # the largest magnitudes regardless of sign, so this must as well.
        rng = random.Random(14)
        n = 24
        basis = orthonormalize(
            [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        )
        planted = [15.0, -11.0, 6.0] + [0.1] * (n - 3)
        diag = [[planted[i] if i == j else 0.0 for j in range(n)]
                for i in range(n)]
        dense = matmul(matmul(basis, diag), transpose(basis))

        values, _ = randomized_eigh(
            dense_to_sparse(dense), k=3, oversample=12,
            power_iterations=4, rng=random.Random(15),
        )
        self.assertAlmostEqual(values[0], 15.0, places=5)
        self.assertAlmostEqual(values[1], -11.0, places=5)
        self.assertAlmostEqual(values[2], 6.0, places=5)

    def test_vectors_reconstruct_the_matrix_at_full_rank(self):
        rng = random.Random(16)
        n = 14
        a = random_symmetric(n, rng)
        values, vectors = randomized_eigh(
            dense_to_sparse(a), k=n, oversample=6,
            power_iterations=3, rng=random.Random(17),
        )
        diag = [[values[i] if i == j else 0.0 for j in range(n)]
                for i in range(n)]
        recon = matmul(matmul(vectors, diag), transpose(vectors))
        for r_a, r_r in zip(a, recon):
            for x, y in zip(r_a, r_r):
                self.assertAlmostEqual(x, y, places=6)

    def test_is_deterministic_for_a_fixed_seed(self):
        rng = random.Random(18)
        a = dense_to_sparse(random_symmetric(12, rng))
        first = randomized_eigh(a, k=4, rng=random.Random(99))
        second = randomized_eigh(a, k=4, rng=random.Random(99))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_k_is_capped_by_matrix_size(self):
        a = dense_to_sparse([[2.0, 0.0], [0.0, 1.0]])
        values, vectors = randomized_eigh(a, k=2, oversample=8)
        self.assertEqual(len(values), 2)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 2)


class TestVectorHelpers(unittest.TestCase):
    def test_unit_has_length_one(self):
        v = unit([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0)
        self.assertAlmostEqual(v[0], 0.6)

    def test_unit_of_zero_stays_zero(self):
        self.assertEqual(unit([0.0, 0.0]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()


class TestBlockKrylov(unittest.TestCase):
    """The Krylov variant must answer the same questions as the other one."""

    def test_recovers_a_planted_spectrum(self):
        rng = random.Random(30)
        n = 30
        basis = orthonormalize(
            [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        )
        planted = [20.0, 12.0, 7.0, 3.0] + [0.2] * (n - 4)
        diag = [[planted[i] if i == j else 0.0 for j in range(n)]
                for i in range(n)]
        dense = matmul(matmul(basis, diag), transpose(basis))
        values, _ = block_krylov_eigh(
            dense_to_sparse(dense), k=4, block=8, depth=4,
            rng=random.Random(31),
        )
        for expected, got in zip(planted[:4], values):
            self.assertAlmostEqual(expected, got, places=5)

    def test_finds_large_negative_eigenvalues_too(self):
        rng = random.Random(32)
        n = 24
        basis = orthonormalize(
            [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        )
        planted = [15.0, -11.0, 6.0] + [0.1] * (n - 3)
        diag = [[planted[i] if i == j else 0.0 for j in range(n)]
                for i in range(n)]
        dense = matmul(matmul(basis, diag), transpose(basis))
        values, _ = block_krylov_eigh(
            dense_to_sparse(dense), k=3, block=8, depth=3,
            rng=random.Random(33),
        )
        self.assertAlmostEqual(values[0], 15.0, places=5)
        self.assertAlmostEqual(values[1], -11.0, places=5)
        self.assertAlmostEqual(values[2], 6.0, places=5)

    def test_vectors_reconstruct_the_matrix_at_full_rank(self):
        rng = random.Random(34)
        n = 14
        a = random_symmetric(n, rng)
        values, vectors = block_krylov_eigh(
            dense_to_sparse(a), k=n, block=7, depth=3, rng=random.Random(35),
        )
        diag = [[values[i] if i == j else 0.0 for j in range(n)]
                for i in range(n)]
        recon = matmul(matmul(vectors, diag), transpose(vectors))
        for r_a, r_r in zip(a, recon):
            for x, y in zip(r_a, r_r):
                self.assertAlmostEqual(x, y, places=6)

    def test_is_deterministic_for_a_fixed_seed(self):
        rng = random.Random(36)
        a = dense_to_sparse(random_symmetric(12, rng))
        first = block_krylov_eigh(a, k=4, block=6, depth=3, rng=random.Random(9))
        second = block_krylov_eigh(a, k=4, block=6, depth=3, rng=random.Random(9))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_eigenpairs_satisfy_the_defining_equation(self):
        rng = random.Random(37)
        n = 12
        a = random_symmetric(n, rng)
        values, vectors = block_krylov_eigh(
            dense_to_sparse(a), k=n, block=6, depth=3, rng=random.Random(38),
        )
        for col, value in enumerate(values):
            v = [row[col] for row in vectors]
            av = [sum(a[i][j] * v[j] for j in range(n)) for i in range(n)]
            for x, y in zip(av, [value * c for c in v]):
                self.assertAlmostEqual(x, y, places=6)

    def test_block_is_capped_by_matrix_size(self):
        a = dense_to_sparse([[2.0, 0.0], [0.0, 1.0]])
        values, vectors = block_krylov_eigh(a, k=2, block=8, depth=2)
        self.assertEqual(len(values), 2)
        self.assertEqual(len(vectors[0]), 2)

    def test_depth_buys_accuracy_on_a_flat_spectrum(self):
        # The mechanism the method exists for. A flat spectrum is what
        # subspace iteration handles badly, because it separates eigenvalues
        # by raising their ratio to a power and the ratios here are all near
        # one. Krylov does not rely on that gap, so going deeper must help.
        #
        # Deliberately not a head-to-head against randomized_eigh: matching
        # the two on work is fiddly (the basis widths differ, and so does the
        # cost of the final projection), and an unmatched comparison would
        # prove nothing. The real comparison is wall-clock on the actual
        # PPMI matrix, which lives in the README.
        rng = random.Random(39)
        n = 40
        basis = orthonormalize(
            [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        )
        planted = [10.0 - 0.1 * i for i in range(n)]
        diag = [[planted[i] if i == j else 0.0 for j in range(n)]
                for i in range(n)]
        sparse = dense_to_sparse(
            matmul(matmul(basis, diag), transpose(basis))
        )

        def error(values):
            return max(abs(abs(a) - b) / b for a, b in zip(values, planted[:8]))

        shallow, _ = block_krylov_eigh(sparse, k=8, block=6, depth=2,
                                       rng=random.Random(40))
        deep, _ = block_krylov_eigh(sparse, k=8, block=6, depth=5,
                                    rng=random.Random(40))
        self.assertLess(error(deep), error(shallow))


class TestPrincipalAngles(unittest.TestCase):
    """Whether two bases disagree about scale or about direction."""

    def test_identical_subspaces_share_everything(self):
        rng = random.Random(50)
        a = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(10)]
        for cosine in principal_angles(a, [row[:] for row in a]):
            self.assertAlmostEqual(cosine, 1.0, places=8)

    def test_a_rescaled_basis_is_the_same_subspace(self):
        # The whole point of the measurement: reweighting the columns cannot
        # change what they span, so this must come back all ones.
        rng = random.Random(51)
        a = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(10)]
        scaled = [[v * s for v, s in zip(row, (5.0, 0.1, 20.0))] for row in a]
        for cosine in principal_angles(a, scaled):
            self.assertAlmostEqual(cosine, 1.0, places=8)

    def test_a_rotated_basis_is_the_same_subspace(self):
        rng = random.Random(52)
        a = [[rng.gauss(0, 1) for _ in range(2)] for _ in range(8)]
        rotated = [[row[0] + row[1], row[0] - row[1]] for row in a]
        for cosine in principal_angles(a, rotated):
            self.assertAlmostEqual(cosine, 1.0, places=8)

    def test_orthogonal_subspaces_share_nothing(self):
        a = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        b = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        for cosine in principal_angles(a, b):
            self.assertAlmostEqual(cosine, 0.0, places=8)

    def test_partial_overlap_is_reported_per_direction(self):
        # One axis shared, one orthogonal: cosines must be 1 and 0, not an
        # average of the two.
        a = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        b = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
        cosines = principal_angles(a, b)
        self.assertAlmostEqual(cosines[0], 1.0, places=8)
        self.assertAlmostEqual(cosines[1], 0.0, places=8)

    def test_a_known_angle(self):
        # Two lines in the plane at 60 degrees; the cosine must be 0.5.
        a = [[1.0], [0.0]]
        b = [[0.5], [math.sqrt(3) / 2]]
        self.assertAlmostEqual(principal_angles(a, b)[0], 0.5, places=8)

    def test_cosines_come_back_sorted(self):
        rng = random.Random(53)
        a = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(12)]
        b = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(12)]
        cosines = principal_angles(a, b)
        self.assertEqual(cosines, sorted(cosines, reverse=True))

    def test_cosines_stay_within_range(self):
        rng = random.Random(54)
        a = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(9)]
        b = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(9)]
        for cosine in principal_angles(a, b):
            self.assertGreaterEqual(cosine, 0.0)
            self.assertLessEqual(cosine, 1.0)

    def test_it_is_symmetric(self):
        rng = random.Random(55)
        a = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(9)]
        b = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(9)]
        for x, y in zip(principal_angles(a, b), principal_angles(b, a)):
            self.assertAlmostEqual(x, y, places=8)
