"""A place where words are points.

By this stage every word is a short list of numbers and nothing else. The
questions you can ask are geometric ones — what is near, what direction
separates these two, where does this land on that axis — and the only
interesting fact about the whole repository is that geometric questions
turn out to have semantic answers.
"""

from __future__ import annotations

import json
import struct
from array import array
from dataclasses import dataclass, field

from .linalg import dot, norm, unit

METRICS = ("cosine", "dot", "euclidean")

_LITTLE_ENDIAN = struct.pack("<H", 1) == struct.pack("=H", 1)


@dataclass
class Space:
    """Word vectors, plus the arithmetic worth doing on them."""

    words: list[str]
    vectors: list[list[float]]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {w: i for i, w in enumerate(self.words)}
        # Cosine similarity is a dot product between unit vectors, and we do
        # a lot of them, so normalise once here rather than in every query.
        self.units = [unit(v) for v in self.vectors]

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    @property
    def dim(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0

    def vector(self, word: str) -> list[float]:
        try:
            return self.vectors[self.index[word]]
        except KeyError:
            raise KeyError(f"{word!r} is not in this space") from None

    def direction(self, word: str) -> list[float]:
        """The unit vector for a word."""
        try:
            return self.units[self.index[word]]
        except KeyError:
            raise KeyError(f"{word!r} is not in this space") from None

    def similarity(self, a: str, b: str, metric: str = "cosine") -> float:
        if metric == "cosine":
            return dot(self.direction(a), self.direction(b))
        if metric == "dot":
            return dot(self.vector(a), self.vector(b))
        if metric == "euclidean":
            return -norm([x - y for x, y in zip(self.vector(a), self.vector(b))])
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")

    def rank(
        self,
        query: list[float],
        n: int = 10,
        exclude: set[str] | None = None,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """The `n` words that best match a query vector.

        `metric` selects how "best" is decided, so the choice can be ablated
        rather than assumed:

        * `cosine` normalises both sides, comparing direction only.
        * `dot` leaves the lengths in, so a long vector scores highly against
          everything. Vector length here tracks word frequency, which is the
          reason to expect this to be bad and the reason to measure it.
        * `euclidean` ranks by negated distance between unnormalised points.
        """
        skip = {self.index[w] for w in (exclude or set()) if w in self.index}
        if metric == "cosine":
            q = unit(query)
            scored = [
                (dot(q, u), i)
                for i, u in enumerate(self.units)
                if i not in skip
            ]
        elif metric == "dot":
            scored = [
                (dot(query, v), i)
                for i, v in enumerate(self.vectors)
                if i not in skip
            ]
        elif metric == "euclidean":
            scored = [
                (-norm([a - b for a, b in zip(query, v)]), i)
                for i, v in enumerate(self.vectors)
                if i not in skip
            ]
        else:
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
        scored.sort(reverse=True)
        return [(self.words[i], s) for s, i in scored[:n]]

    def neighbors(
        self, word: str, n: int = 10, metric: str = "cosine"
    ) -> list[tuple[str, float]]:
        """The words closest to `word`, excluding the word itself."""
        return self.rank(self.vector(word), n=n, exclude={word}, metric=metric)

    def analogy(
        self, a: str, b: str, c: str, n: int = 5, metric: str = "cosine"
    ) -> list[tuple[str, float]]:
        """`a` is to `b` as `c` is to what?

        The classic vector-offset method: whatever transformation carries `a`
        to `b` is assumed to be a translation, so applying it to `c` should
        land near the answer. The three input words are excluded from the
        result, because otherwise they win — they are, after all, the three
        points the query was built out of, and a query built from `king`
        is nearer to `king` than to anything else.
        """
        # The offset has to be built in whatever space the comparison
        # happens in. Building it from unit vectors and then ranking by a
        # metric that cares about length gives a target of magnitude ~1
        # against candidates of magnitude ~100, so every candidate is about
        # equally far away and the winner is just whichever vector is
        # shortest. That is an artefact of mixing the two spaces, not a fact
        # about the metric, and it scored a clean 0% until it was fixed.
        source = self.direction if metric == "cosine" else self.vector
        va, vb, vc = source(a), source(b), source(c)
        target = [y - x + z for x, y, z in zip(va, vb, vc)]
        return self.rank(target, n=n, exclude={a, b, c}, metric=metric)

    def analogy_mul(
        self, a: str, b: str, c: str, n: int = 5, epsilon: float = 1e-3
    ) -> list[tuple[str, float]]:
        """`a` is to `b` as `c` is to what, scored multiplicatively.

        Additive analogy sums three similarities, so one large term can carry
        the answer on its own — which is why vector-offset analogies so often
        return a word that is merely very close to `c` and ignores `b`. The
        multiplicative form asks for all three conditions at once:

            argmax   sim(d, b) * sim(d, c) / (sim(d, a) + epsilon)

        A product cannot be rescued by one strong factor; every term has to
        be satisfied. Cosines are mapped from [-1, 1] to [0, 1] first, since
        a product of signed quantities would let two negatives agree.

        This is Levy and Goldberg's 3CosMul, and the reason it is here is
        that it is reported to help most exactly where this corpus lives:
        at small scale, where the additive form is least reliable.
        """
        da, db, dc = self.direction(a), self.direction(b), self.direction(c)
        skip = {a, b, c}
        scored = []
        for i, u in enumerate(self.units):
            if self.words[i] in skip:
                continue
            pa = (dot(u, da) + 1.0) / 2.0
            pb = (dot(u, db) + 1.0) / 2.0
            pc = (dot(u, dc) + 1.0) / 2.0
            scored.append((pb * pc / (pa + epsilon), i))
        scored.sort(reverse=True)
        return [(self.words[i], s) for s, i in scored[:n]]

    def axis(
        self, negative: str, positive: str, words: list[str]
    ) -> list[tuple[str, float]]:
        """Project words onto the line running from one word to another.

        Useful for seeing whether a contrast the corpus never states outright
        exists in the geometry anyway. Results are sorted from the `negative`
        end to the `positive` end.
        """
        a, b = self.direction(negative), self.direction(positive)
        line = unit([y - x for x, y in zip(a, b)])
        scored = [
            (w, dot(self.direction(w), line)) for w in words if w in self
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored

    # ----------------------------------------------------------------
    # variations on the same factorisation
    # ----------------------------------------------------------------

    def _eigenvalues(self) -> list[float]:
        values = self.meta.get("eigenvalues")
        if not values or len(values) != self.dim:
            raise ValueError(
                "this space has no recorded eigenvalues, so it cannot be "
                "reweighted or sliced; rebuild it with sens.pipeline.build"
            )
        return list(values)

    def rescaled(self, power: float) -> "Space":
        """The same factorisation with a different eigenvalue weighting.

        Word vectors are eigenvectors scaled by `|eigenvalue| ** power`, and
        that exponent is a free parameter nobody derives from anything — 0,
        0.5 and 1 all appear in the literature. Changing it does not require
        touching the corpus, so sweeping it costs one evaluation each rather
        than one full rebuild each.
        """
        values = self._eigenvalues()
        current = self.meta.get("config", {}).get("eigenvalue_power", 0.5)
        delta = power - current
        factors = [
            (abs(v) ** delta) if v else 0.0 for v in values
        ]
        meta = dict(self.meta)
        meta["config"] = {**meta.get("config", {}), "eigenvalue_power": power}
        return Space(
            words=list(self.words),
            vectors=[
                [x * f for x, f in zip(row, factors)] for row in self.vectors
            ],
            meta=meta,
        )

    def shrunk(self, threshold: float) -> "Space":
        """Soft-threshold the eigenvalues before they weight the axes.

        Every eigenvalue moves toward zero by `threshold`, and any that would
        cross it is set there:

            |lambda|  ->  max(0, |lambda| - threshold)

        This exists because of a measured accident. An inaccurate
        factorisation systematically underestimates the poorly determined
        tail, and that turned out to *help* — the tail is mostly noise, and
        under-weighting it is regularisation arrived at by mistake. Soft
        thresholding is the deliberate version: keep the accurate spectrum,
        then discount it on purpose, by an amount you can tune and report
        rather than one that falls out of how many iterations you happened
        to run.
        """
        values = self._eigenvalues()
        power = self.meta.get("config", {}).get("eigenvalue_power", 1.0)
        factors = []
        for value in values:
            magnitude = abs(value)
            if magnitude <= 0.0:
                factors.append(0.0)
                continue
            kept = max(0.0, magnitude - threshold)
            factors.append((kept / magnitude) ** power)
        meta = dict(self.meta)
        meta["config"] = {**meta.get("config", {}), "shrinkage": threshold}
        return Space(
            words=list(self.words),
            vectors=[
                [x * f for x, f in zip(row, factors)] for row in self.vectors
            ],
            meta=meta,
        )

    def subspace(self, keep: list[int]) -> "Space":
        """A space using only the listed dimensions.

        The point of this is ablation. A dimension can be removed and the
        result measured, which is the only way to find out whether an axis
        was carrying anything.
        """
        values = self._eigenvalues()
        meta = dict(self.meta)
        meta["eigenvalues"] = [values[i] for i in keep]
        return Space(
            words=list(self.words),
            vectors=[[row[i] for i in keep] for row in self.vectors],
            meta=meta,
        )

    def positive_dimensions(self) -> list[int]:
        """Indices whose eigenvalue is positive."""
        return [i for i, v in enumerate(self._eigenvalues()) if v > 0]

    def negative_dimensions(self) -> list[int]:
        return [i for i, v in enumerate(self._eigenvalues()) if v < 0]

    # ----------------------------------------------------------------
    # persistence
    # ----------------------------------------------------------------

    MAGIC = b"SENS0001"

    def save(self, path: str) -> None:
        """Write the space to a small binary file.

        Header is JSON so the file explains itself; the vectors are raw
        little-endian float32, because 4 bytes is more precision than a
        cosine ranking can use and the file is a build artifact anyway.
        """
        header = json.dumps(
            {
                "words": self.words,
                "dim": self.dim,
                "meta": self.meta,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        flat = array("f", (x for v in self.vectors for x in v))
        if not _LITTLE_ENDIAN:
            flat.byteswap()
        with open(path, "wb") as handle:
            handle.write(self.MAGIC)
            handle.write(struct.pack("<I", len(header)))
            handle.write(header)
            handle.write(flat.tobytes())

    @classmethod
    def load(cls, path: str) -> "Space":
        with open(path, "rb") as handle:
            if handle.read(len(cls.MAGIC)) != cls.MAGIC:
                raise ValueError(f"{path} is not a sens space file")
            (size,) = struct.unpack("<I", handle.read(4))
            header = json.loads(handle.read(size).decode("utf-8"))
            flat = array("f")
            flat.frombytes(handle.read())
        if not _LITTLE_ENDIAN:
            flat.byteswap()
        dim = header["dim"]
        words = header["words"]
        vectors = [
            list(flat[i * dim : (i + 1) * dim]) for i in range(len(words))
        ]
        return cls(words=words, vectors=vectors, meta=header.get("meta", {}))
