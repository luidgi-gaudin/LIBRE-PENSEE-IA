"""Re-derive the register's numbers from the corpus and check they match.

`sens claims` audits whether a claim has the controls its status requires.
That is a question about method. This asks the other one, which no amount of
methodological care answers: **is the number still true?**

Nothing in this repository catches a digit transposed while writing a table,
or a refactor that quietly moves a result by a standard deviation. Both are
silent, both are the kind of thing that survives review, and neither is the
sort of mistake trying harder prevents. So the claims carry the recipe for
their own re-derivation, and this runs it.

The tolerance is the claim's own noise floor. A rebuild starts from a
different random block, so a verified effect will not reproduce exactly; it
should reproduce within the spread that identical rebuilds show. Anything
outside that is either the code moving or the register being wrong, and
either way somebody should look.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .audit import Ruler, build_ruler
from .claims import REGISTER, Claim
from .noise import DEFAULT_SEEDS
from .pipeline import Config

# The measured spread of held-out Spearman across identical rebuilds. A
# re-derived effect is a difference of two such numbers, so it inherits
# roughly twice the variance; the tolerance is set accordingly rather than
# optimistically.
# The tolerance for *re-derivation* is not the same quantity as the noise
# floor a *claim* is judged against, and conflating them would be an error.
# Verification re-runs with every seed fixed, so it reproduces exactly; this
# margin exists to catch code drift, not sampling. Judging claims uses
# claims.EFFECT_SD, which is more than twice as wide because it counts the
# choices verification deliberately holds still.
HELDOUT_SD = 0.0047
TOLERANCE = {"spearman": 4 * HELDOUT_SD}


@dataclass
class Result:
    """One claim, re-measured."""

    claim: Claim
    observed: float
    seconds: float
    collection: str = "novels"

    @property
    def recorded(self) -> float:
        return self.claim.recorded(self.collection) or 0.0

    @property
    def drift(self) -> float:
        return self.observed - self.recorded

    @property
    def tolerance(self) -> float:
        return TOLERANCE.get(self.claim.unit, float("inf"))

    @property
    def agrees(self) -> bool:
        return abs(self.drift) <= self.tolerance

    @property
    def row(self) -> str:
        mark = "ok " if self.agrees else "DRIFT"
        return (
            f"{self.claim.id:<22} recorded {self.recorded:+.4f}  "
            f"observed {self.observed:+.4f}  drift {self.drift:+.4f}  "
            f"{mark:<5} {self.seconds:5.1f}s"
        )


def config_delta(ruler: Ruler, claim: Claim, base: Config) -> float:
    """The effect of moving one parameter away from its default."""
    recipe = claim.check
    default = ruler.score(base)
    alternative = ruler.score(
        Config(**{**base.as_dict(), recipe.parameter: recipe.against})
    )
    return default - alternative


def verify(
    paths: list[str],
    only: tuple[str, ...] = (),
    base: Config | None = None,
    progress=None,
    collection: str = "novels",
) -> list[Result]:
    """Re-derive every verifiable claim, or just the named ones.

    `collection` selects which recorded number to check against. The
    replication figures were measured once, by hand, in a terminal, and
    lived nowhere the code could see — exactly the situation that let the
    novels column go stale for six commits.
    """
    base = base or Config()
    wanted = [
        c for c in REGISTER
        if c.checkable_on(collection) and (not only or c.id in only)
    ]
    if not wanted:
        return []

    ruler = build_ruler(paths)
    # The default build is shared by every comparison, so score it once.
    baseline = ruler.score(base)

    results = []
    for claim in wanted:
        if progress:
            progress(claim)
        start = time.time()
        recipe = claim.check
        alternative = ruler.score(
            Config(**{**base.as_dict(), recipe.parameter: recipe.against})
        )
        results.append(
            Result(
                claim=claim,
                observed=baseline - alternative,
                seconds=time.time() - start,
                collection=collection,
            )
        )
    return results


def header() -> str:
    return (
        f"{'claim':<22} {'recorded':>14}  {'observed':>15}  "
        f"{'drift':>13}  {'':<5} {'time':>6}"
    )


def summarise(results: list[Result]) -> str:
    drifted = [r for r in results if not r.agrees]
    if not results:
        return "nothing to verify"
    if not drifted:
        return (
            f"all {len(results)} re-derived within tolerance "
            f"(±{TOLERANCE['spearman']:.4f}, four times the rebuild spread)"
        )
    names = ", ".join(r.claim.id for r in drifted)
    return (
        f"{len(drifted)} of {len(results)} drifted beyond tolerance: {names}\n"
        "either the code moved or the register is wrong; both need a look"
    )
