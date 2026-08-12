"""Enough linear algebra to factor a matrix, and not one line more.

There is no NumPy here, on purpose. Every multiply-add in this file is a
Python float operation you could step through in a debugger. The point of
the repository is that the machinery is small enough to hold in your head,
and importing an opaque `svd()` would hide exactly the step that matters.

Dense matrices are lists of rows; each row is a list of floats. Sparse
matrices are lists of `{column: value}` dicts, one per row.

The recurring performance idiom is

    acc = [a + s * b for a, b in zip(acc, row)]

which is a scaled vector addition (an `axpy`) written so CPython spends its
time in the zip/listcomp fast path instead of in index arithmetic. Almost
all the running time of this module is inside that one line.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

Dense = list[list[float]]


# --------------------------------------------------------------------------
# sparse
# --------------------------------------------------------------------------


class SparseMatrix:
    """A square sparse matrix, stored by rows."""

    __slots__ = ("rows", "n")

    def __init__(self, rows: list[dict[int, float]]):
        self.rows = rows
        self.n = len(rows)

    @property
    def nnz(self) -> int:
        return sum(len(r) for r in self.rows)

    def dot_dense(self, x: Dense) -> Dense:
        """Return `self @ x` for a dense `n x k` right-hand side."""
        k = len(x[0]) if x else 0
        out: Dense = []
        append = out.append
        for row in self.rows:
            acc = [0.0] * k
            for j, v in row.items():
                xj = x[j]
                acc = [a + v * b for a, b in zip(acc, xj)]
            append(acc)
        return out

    def to_dense(self) -> Dense:
        dense = [[0.0] * self.n for _ in range(self.n)]
        for i, row in enumerate(self.rows):
            di = dense[i]
            for j, v in row.items():
                di[j] = v
        return dense


# --------------------------------------------------------------------------
# small dense helpers
# --------------------------------------------------------------------------


def gaussian(n: int, k: int, rng: random.Random) -> Dense:
    """An `n x k` matrix of standard normal noise."""
    return [[rng.gauss(0.0, 1.0) for _ in range(k)] for _ in range(n)]


def gram(x: Dense) -> Dense:
    """Return the `k x k` matrix `x.T @ x`."""
    k = len(x[0]) if x else 0
    g = [[0.0] * k for _ in range(k)]
    for row in x:
        for a, ra in enumerate(row):
            if ra:
                ga = g[a]
                g[a] = [v + ra * rb for v, rb in zip(ga, row)]
    return g


def cross(a: Dense, b: Dense) -> Dense:
    """Return `a.T @ b` for two matrices with the same number of rows."""
    ka = len(a[0]) if a else 0
    kb = len(b[0]) if b else 0
    out = [[0.0] * kb for _ in range(ka)]
    for ra, rb in zip(a, b):
        for i, v in enumerate(ra):
            if v:
                oi = out[i]
                out[i] = [o + v * w for o, w in zip(oi, rb)]
    return out


def apply_right(x: Dense, m: Dense) -> Dense:
    """Return `x @ m`, accumulating one row of `m` at a time."""
    k = len(m[0]) if m else 0
    out: Dense = []
    for row in x:
        acc = [0.0] * k
        for v, mrow in zip(row, m):
            if v:
                acc = [a + v * b for a, b in zip(acc, mrow)]
        out.append(acc)
    return out


def cholesky(a: Dense) -> Dense:
    """Lower-triangular `L` with `L @ L.T == a`, for symmetric positive `a`.

    A tiny ridge is added to the diagonal when a pivot goes non-positive.
    That happens when the columns we are orthonormalising are nearly
    dependent, which is common in the last few columns of a random sketch,
    and the ridge keeps the factorisation going rather than raising.
    """
    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(lower[i][t] * lower[j][t] for t in range(j))
            if i == j:
                pivot = a[i][i] - s
                if pivot <= 1e-12:
                    pivot = 1e-12
                lower[i][j] = math.sqrt(pivot)
            else:
                lower[i][j] = (a[i][j] - s) / lower[j][j]
    return lower


def invert_lower(lower: Dense) -> Dense:
    """Invert a lower-triangular matrix by forward substitution."""
    n = len(lower)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        inv[i][i] = 1.0 / lower[i][i]
        for j in range(i):
            s = sum(lower[i][t] * inv[t][j] for t in range(j, i))
            inv[i][j] = -s / lower[i][i]
    return inv


def orthonormalize(x: Dense) -> Dense:
    """Return a matrix with orthonormal columns spanning the columns of `x`.

    This is CholeskyQR run twice. One pass is `Q = X (L^-1).T` where `L` comes
    from the Cholesky factor of `X.T X`; that is exact in real arithmetic but
    loses roughly the square of the condition number in floating point. Running
    it a second time on the already-improved `Q` recovers the lost digits.
    Two cheap passes beat one careful Gram-Schmidt here because both passes are
    built out of the same `axpy` idiom the rest of this file runs on.
    """
    for _ in range(2):
        g = gram(x)
        rinv_t = _transpose(invert_lower(cholesky(g)))
        x = apply_right(x, rinv_t)
    return x


def _transpose(m: Dense) -> Dense:
    return [list(col) for col in zip(*m)]


# --------------------------------------------------------------------------
# eigendecomposition
# --------------------------------------------------------------------------


def jacobi_eigh(matrix: Dense, sweeps: int = 60, tol: float = 1e-12):
    """Eigendecompose a small symmetric matrix by cyclic Jacobi rotations.

    Returns `(values, vectors)` with eigenvalues in descending order and
    eigenvectors as columns of `vectors`.

    Jacobi works by repeatedly picking an off-diagonal entry and applying the
    plane rotation that zeroes it. Each rotation disturbs the entries zeroed
    before it, but never by as much as it removes, so sweeping over every pair
    in turn drives the off-diagonal mass to zero. It is not the fastest
    algorithm for large matrices; at the size we use it (a few dozen rows) it
    is fast enough and it is the one that fits in a screen of code.
    """
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(sweeps):
        off = math.sqrt(
            sum(a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n))
        )
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= tol:
                    continue
                # theta parameterises the rotation that annihilates a[p][q].
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq

    values = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -values[i])
    sorted_values = [values[i] for i in order]
    sorted_vectors = [[row[i] for i in order] for row in v]
    return sorted_values, sorted_vectors


def randomized_eigh(
    matrix: SparseMatrix,
    k: int,
    oversample: int = 16,
    power_iterations: int = 3,
    rng: random.Random | None = None,
):
    """Approximate the `k` dominant eigenpairs of a large symmetric matrix.

    Returns `(values, vectors)` where `vectors` has one row per matrix row and
    `k` columns, ordered by descending `|value|`.

    The method is the standard randomised range finder. Multiply the matrix by
    a block of random vectors; the result leans towards the directions the
    matrix stretches most, because that is what a matrix does to noise. A few
    extra multiplications sharpen the lean, since each one raises the ratio
    between any two eigenvalues to a higher power and so pushes the ones we
    do not want further down. Orthonormalising the block gives a small basis
    `Q` in which the matrix is only `p x p`, and that one we decompose exactly.

    Sampling more columns than we need (`oversample`) is what makes this
    reliable rather than merely lucky: the extra directions absorb the error
    that would otherwise contaminate the ones we keep.
    """
    rng = rng or random.Random(0)
    n = matrix.n
    p = min(n, k + oversample)

    q = orthonormalize(matrix.dot_dense(gaussian(n, p, rng)))
    for _ in range(power_iterations):
        # A is symmetric, so `A.T A = A A` and one product per iteration
        # does the work that a general randomised SVD needs two for.
        q = orthonormalize(matrix.dot_dense(q))

    aq = matrix.dot_dense(q)
    small = cross(q, aq)  # Q.T A Q, the projection of A into the basis
    # Q.T A Q is symmetric in exact arithmetic; averaging with its transpose
    # removes the rounding drift so Jacobi gets the input it expects.
    size = len(small)
    small = [
        [(small[i][j] + small[j][i]) * 0.5 for j in range(size)]
        for i in range(size)
    ]

    values, vectors = jacobi_eigh(small)
    order = sorted(range(len(values)), key=lambda i: -abs(values[i]))[:k]
    kept_values = [values[i] for i in order]
    kept_vectors = [[row[i] for i in order] for row in vectors]

    return kept_values, apply_right(q, kept_vectors)


def _concat(blocks: list[Dense]) -> Dense:
    """Glue blocks together side by side into one wide matrix."""
    return [
        [value for block in blocks for value in block[row]]
        for row in range(len(blocks[0]))
    ]


def block_krylov_eigh(
    matrix: SparseMatrix,
    k: int,
    block: int = 24,
    depth: int = 4,
    rng: random.Random | None = None,
):
    """Approximate the `k` dominant eigenpairs, keeping every iterate.

    Same shape of answer as `randomized_eigh`, and better at getting it, for
    the reason that method struggles: subspace iteration computes
    `A, A^2, ..., A^q` applied to a random block and then throws all of them
    away except the last. Everything it discarded was information about the
    matrix. Keeping the whole sequence

        K = [ A W , A^2 W , ... , A^q W ]

    and searching that span instead is the block Krylov method, and it costs
    the same matrix multiplications.

    The difference shows up precisely where a PPMI matrix lives. Subspace
    iteration separates eigenvalues by raising their ratio to a power, so it
    works well when the spectrum drops off sharply and badly when it is flat.
    A Krylov space does not rely on that gap: it can represent any polynomial
    in `A` applied to the starting block, and the best polynomial for
    isolating a flat cluster is much better than `x^q`.

    Cost is set by the width of the basis, `block * depth`, so a narrow block
    taken deep buys accuracy that a wide block taken shallow cannot.
    """
    rng = rng or random.Random(0)
    n = matrix.n
    block = min(block, n)

    blocks: list[Dense] = []
    y = gaussian(n, block, rng)
    for _ in range(depth):
        # Orthonormalising between steps is not optional. Without it every
        # block collapses toward the dominant eigenvector in floating point
        # and the later ones carry no independent information.
        y = orthonormalize(matrix.dot_dense(y))
        blocks.append(y)

    q = orthonormalize(_concat(blocks))
    aq = matrix.dot_dense(q)
    small = cross(q, aq)
    size = len(small)
    small = [
        [(small[i][j] + small[j][i]) * 0.5 for j in range(size)]
        for i in range(size)
    ]

    values, vectors = jacobi_eigh(small)
    order = sorted(range(len(values)), key=lambda i: -abs(values[i]))[:k]
    kept_values = [values[i] for i in order]
    kept_vectors = [[row[i] for i in order] for row in vectors]
    return kept_values, apply_right(q, kept_vectors)


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def unit(v: Sequence[float]) -> list[float]:
    n = norm(v)
    if n == 0.0:
        return [0.0] * len(v)
    return [x / n for x in v]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
