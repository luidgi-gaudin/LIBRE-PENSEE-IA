"""The whole method, end to end, in one function.

    text  ->  tokens  ->  counts  ->  surprise  ->  factorisation  ->  space

Five steps. The first three are bookkeeping. The fourth is the only one that
makes a claim about language. The fifth is the one that produces meaning,
and it does so by throwing information away.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

from .counts import cooccurrence
from .space import Space
from .text import Vocabulary, read_tokens
from .weight import ppmi
from .linalg import block_krylov_eigh, randomized_eigh


@dataclass
class Config:
    """Every knob, in one place, so a build is reproducible from a dict."""

    vocab_size: int = 4000
    min_count: int = 10
    window: int = 4
    harmonic: bool = True
    min_pair_weight: float = 1.0
    alpha: float = 1.0
    shift: float = 1.0
    dim: int = 64
    factoriser: str = "subspace"
    krylov_block: int = 24
    krylov_depth: int = 4
    oversample: int = 16
    power_iterations: int = 3
    eigenvalue_power: float = 1.0
    seed: int = 20260811

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Report:
    """What the build saw on the way through. Printed, not persisted."""

    stages: list[tuple[str, str, float]] = field(default_factory=list)

    def record(self, name: str, detail: str, seconds: float) -> None:
        self.stages.append((name, detail, seconds))

    def render(self) -> str:
        width = max((len(s[0]) for s in self.stages), default=0)
        return "\n".join(
            f"  {name:<{width}}  {detail:<34} {seconds:6.1f}s"
            for name, detail, seconds in self.stages
        )


def build(
    paths: Iterable[str],
    config: Config | None = None,
    verbose: bool = True,
) -> tuple[Space, Report]:
    """Run the pipeline over a list of text files."""
    config = config or Config()
    report = Report()
    paths = list(paths)

    def stage(name: str):
        start = time.time()

        def done(detail: str) -> None:
            elapsed = time.time() - start
            report.record(name, detail, elapsed)
            if verbose:
                print(
                    f"  {name:<14} {detail:<34} {elapsed:6.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

        return done

    done = stage("read")
    tokens: list[str] = []
    for path in paths:
        tokens.extend(read_tokens(path))
    done(f"{len(tokens):,} tokens from {len(paths)} file(s)")

    done = stage("vocabulary")
    vocab = Vocabulary.from_tokens(
        tokens, max_size=config.vocab_size, min_count=config.min_count
    )
    ids = vocab.encode(tokens)
    coverage = len(ids) / max(1, len(tokens))
    done(f"{len(vocab):,} types, {coverage:.1%} of tokens kept")

    done = stage("co-occurrence")
    counts = cooccurrence(
        ids,
        size=len(vocab),
        window=config.window,
        harmonic=config.harmonic,
        min_weight=config.min_pair_weight,
    )
    density = counts.nnz / max(1, len(vocab) ** 2)
    done(f"{counts.nnz:,} pairs, {density:.1%} dense")

    done = stage("ppmi")
    matrix = ppmi(counts, alpha=config.alpha, shift=config.shift)
    kept = matrix.nnz / max(1, counts.nnz)
    done(f"{matrix.nnz:,} positive, {kept:.1%} of pairs")

    done = stage("factorise")
    if config.factoriser == "krylov":
        values, vectors = block_krylov_eigh(
            matrix,
            k=config.dim,
            block=config.krylov_block,
            depth=config.krylov_depth,
            rng=random.Random(config.seed),
        )
    else:
        values, vectors = randomized_eigh(
            matrix,
            k=config.dim,
            oversample=config.oversample,
            power_iterations=config.power_iterations,
            rng=random.Random(config.seed),
        )
    spread = abs(values[0]) / max(abs(values[-1]), 1e-12)
    done(f"{config.dim} dims, eigenvalue spread {spread:6.1f}x")

    # A = U L U.T for a symmetric A, so the truncated SVD's left factor is
    # U scaled by |L| to some power. The usual choice is 0.5, which splits
    # the singular values evenly between the word side and the context side.
    #
    # This corpus disagrees, and it took two independent measurements to be
    # sure of it. `sens sweep` finds morphological analogy peaking between
    # 0.75 and 1.0; `sens heldout` finds prediction of unseen co-occurrence
    # peaking at 1.0. Both curves turn over after their maximum rather than
    # running to the edge of the range, which is what rules out the obvious
    # objection — that a metric was simply rewarding whichever exponent
    # reconstructs the fitted matrix best. Correlation against the *fitted*
    # matrix peaks at 0.75 and then falls, so the instrument is not just
    # measuring reconstruction.
    scale = [abs(v) ** config.eigenvalue_power for v in values]
    scaled = [[x * s for x, s in zip(row, scale)] for row in vectors]

    space = Space(
        words=list(vocab.words),
        vectors=scaled,
        meta={
            "config": config.as_dict(),
            "sources": paths,
            "tokens": len(tokens),
            "eigenvalues": [round(v, 4) for v in values],
        },
    )
    return space, report
