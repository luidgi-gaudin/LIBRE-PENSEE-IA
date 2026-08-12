"""Does the compression generalise, or only memorise?

The analogy benchmark measures one thing, and the pipeline has a setting the
analogy benchmark alone cannot arbitrate: raising the eigenvalue exponent
helps analogy, and analogy is a global linear structure, so a metric built
only from analogies is biased toward exactly the change it is being asked to
judge. Deciding on that evidence would be circular.

This is the second measurement. Split the corpus in half. Build the space on
one half. Then compute, from the *other* half, a ground-truth similarity
between words — the cosine between their raw uncompressed PPMI rows — and ask
how well the 64-dimensional space predicts it.

The claim being tested is the one the whole repository rests on: that
squeezing co-occurrence through too few dimensions keeps the structure and
discards the accidents. If that is true, a space built on half a corpus
should predict similarities measured on text it has never read. If it is
false, the space memorised its own half and the number will say so.

Why cosine against cosine, rather than reconstructing PPMI values directly:
a dot-product reconstruction is exactly right at exponent 0.5 and wrong
everywhere else by construction, which would smuggle the answer into the
instrument. Comparing cosines has no such built-in preference, and cosine is
what the space is actually used for.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .counts import cooccurrence
from .linalg import SparseMatrix
from .pipeline import Config, build
from .space import Space
from .text import Vocabulary, read_tokens
from .weight import ppmi


# The ruler, frozen.
#
# The ground truth is a PPMI table, so it has all the same parameters the
# model has. Deriving it from the model's config seemed natural and was a
# mistake: changing the `alpha` default moved every held-out number by 0.14
# at once, because the space and the yardstick moved together. Nothing had
# got worse; the measurement had simply been redefined underneath itself.
#
# So these values are pinned here and do not track `Config`. They are one
# fixed operationalisation of "how similar are these two words in text the
# model has not read". The absolute numbers depend on this choice; the
# comparisons between builds, which is what the measurement is for, do not.
RULER = {
    "window": 4,
    "harmonic": True,
    "min_pair_weight": 1.0,
    "alpha": 0.75,
    "shift": 1.0,
}


def split_documents(
    documents: list[list[str]], block: int = 4000
) -> tuple[list[str], list[str]]:
    """Split each document independently, then concatenate.

    `split_blocks` alternates over one stream, so where a document boundary
    falls decides the phase of every block after it — and therefore which
    half of the corpus a token lands in. That made the held-out split depend
    on the order the files happened to be listed in, which is incidental,
    and `sens verify` measured the sensitivity at 5 to 6 standard deviations:
    larger than the random seed, from a choice nobody thought they were
    making.

    Splitting per document removes it. A document contributes the same
    blocks to train and to test wherever it sits in the list. The
    concatenation order still varies, but that only affects adjacency across
    the handful of joins, not membership.
    """
    train: list[str] = []
    test: list[str] = []
    for document in documents:
        left, right = split_blocks(document, block=block)
        train.extend(left)
        test.extend(right)
    return train, test


def split_blocks(
    tokens: list[str], block: int = 4000
) -> tuple[list[str], list[str]]:
    """Deal the corpus into two halves, alternating in blocks.

    Not a straight cut down the middle. These are novels: the first half of
    `Moby-Dick` is Nantucket and the second half is the chase, so a single
    split would test the model on vocabulary and subject matter the training
    half barely contains, and would measure topic drift rather than
    generalisation. Alternating blocks keeps both halves representative
    while still making them disjoint text.

    The blocks are large enough that no co-occurrence window spans the seam
    more than negligibly: at 4,000 tokens per block and a window of 4, one
    boundary in every thousand positions is affected.
    """
    train: list[str] = []
    test: list[str] = []
    for start in range(0, len(tokens), block):
        chunk = tokens[start : start + block]
        if (start // block) % 2 == 0:
            train.extend(chunk)
        else:
            test.extend(chunk)
    return train, test


def sparse_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Cosine between two sparse rows."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    shared = sum(v * b[k] for k, v in a.items() if k in b)
    if shared == 0.0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return shared / (na * nb)


def _ranks(values: list[float]) -> list[float]:
    """Ranks with ties averaged, as Spearman requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation.

    Rank rather than value, because the space is only ever used to *order*
    words — nearest neighbours, ranked analogy answers. Whether it reproduces
    the magnitude of a similarity is not a question anyone asks of it.
    """
    return pearson(_ranks(xs), _ranks(ys))


@dataclass
class Prepared:
    """Everything needed to score a space against held-out text.

    Built once. The exponent sweep then costs one correlation each, because
    reweighting does not touch the corpus.
    """

    space: Space
    truth: SparseMatrix
    vocab: Vocabulary
    pairs: list[tuple[int, int]]
    train_tokens: int
    test_tokens: int

    @property
    def coverage(self) -> float:
        return len(self.pairs) / max(1, len(self.vocab))


def prepare(
    paths: list[str],
    config: Config | None = None,
    block: int = 4000,
    pair_vocab: int = 1200,
    pair_count: int = 4000,
    seed: int = 20260811,
    verbose: bool = False,
    truth_config: Config | None = None,
) -> Prepared:
    """Build a space on one half of the corpus and a truth table on the other.

    `truth_config` overrides how the ground truth is computed. It defaults to
    the frozen `RULER` above rather than to `config`, so that changing a
    pipeline default cannot silently redefine the measurement.
    """
    config = config or Config()
    truth_config = truth_config or Config(**{**config.as_dict(), **RULER})

    documents = [read_tokens(path) for path in paths]
    tokens = [t for d in documents for t in d]
    train, test = split_documents(documents, block=block)

    space, _ = build_from_tokens(train, config, verbose=verbose)

    # The truth table uses the training vocabulary, because a word the space
    # has no vector for cannot be scored either way.
    vocab = Vocabulary(
        words=list(space.words),
        counts=[1] * len(space.words),
    )
    truth = ppmi(
        cooccurrence(
            vocab.encode(test),
            size=len(vocab),
            window=truth_config.window,
            harmonic=truth_config.harmonic,
            min_weight=truth_config.min_pair_weight,
        ),
        alpha=truth_config.alpha,
        shift=truth_config.shift,
    )

    # Pairs are drawn from the frequent end of the vocabulary. A rare word's
    # held-out row is mostly sampling noise, so including it would measure
    # the corpus's thinness rather than the model's generalisation.
    ceiling = min(pair_vocab, len(vocab))
    rng = random.Random(seed)
    candidates = set()
    attempts = 0
    while len(candidates) < pair_count and attempts < pair_count * 20:
        attempts += 1
        i = rng.randrange(ceiling)
        j = rng.randrange(ceiling)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in candidates:
            continue
        if not truth.rows[key[0]] or not truth.rows[key[1]]:
            continue
        candidates.add(key)

    return Prepared(
        space=space,
        truth=truth,
        vocab=vocab,
        pairs=sorted(candidates),
        train_tokens=len(train),
        test_tokens=len(test),
    )


def build_from_tokens(
    tokens: list[str], config: Config, verbose: bool = False
) -> tuple[Space, object]:
    """Run the pipeline on an in-memory token list.

    `pipeline.build` reads files. Held-out evaluation needs to build from a
    split that only exists in memory, so this writes the tokens nowhere and
    calls the same stages.
    """
    import tempfile
    import os

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    )
    try:
        handle.write(" ".join(tokens))
        handle.close()
        return build([handle.name], config, verbose=verbose)
    finally:
        os.unlink(handle.name)


def correlation(space: Space, prepared: Prepared) -> tuple[float, int]:
    """Spearman between predicted similarity and held-out similarity."""
    predicted = []
    actual = []
    for i, j in prepared.pairs:
        truth = sparse_cosine(prepared.truth.rows[i], prepared.truth.rows[j])
        if truth == 0.0:
            continue
        predicted.append(
            sum(
                x * y
                for x, y in zip(space.units[i], space.units[j])
            )
        )
        actual.append(truth)
    return spearman(predicted, actual), len(predicted)


def sweep_power(
    prepared: Prepared, powers: tuple[float, ...]
) -> list[tuple[float, float, int]]:
    """Score several eigenvalue exponents against the held-out truth."""
    results = []
    for power in powers:
        rho, used = correlation(prepared.space.rescaled(power), prepared)
        results.append((power, rho, used))
    return results
