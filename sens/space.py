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

from .linalg import dot, unit

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

    def similarity(self, a: str, b: str) -> float:
        return dot(self.direction(a), self.direction(b))

    def rank(
        self,
        query: list[float],
        n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """The `n` words whose directions best match a query vector."""
        q = unit(query)
        skip = {self.index[w] for w in (exclude or set()) if w in self.index}
        scored = [
            (dot(q, u), i)
            for i, u in enumerate(self.units)
            if i not in skip
        ]
        scored.sort(reverse=True)
        return [(self.words[i], s) for s, i in scored[:n]]

    def neighbors(self, word: str, n: int = 10) -> list[tuple[str, float]]:
        """The words closest to `word`, excluding the word itself."""
        return self.rank(self.vector(word), n=n, exclude={word})

    def analogy(
        self, a: str, b: str, c: str, n: int = 5
    ) -> list[tuple[str, float]]:
        """`a` is to `b` as `c` is to what?

        The classic vector-offset method: whatever transformation carries `a`
        to `b` is assumed to be a translation, so applying it to `c` should
        land near the answer. The three input words are excluded from the
        result, because otherwise they win — they are, after all, the three
        points the query was built out of, and a query built from `king`
        is nearer to `king` than to anything else.
        """
        va, vb, vc = self.direction(a), self.direction(b), self.direction(c)
        target = [y - x + z for x, y, z in zip(va, vb, vc)]
        return self.rank(target, n=n, exclude={a, b, c})

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
