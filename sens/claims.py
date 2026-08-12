"""Every claim this repository makes, as data that can be checked.

The most useful thing here has not been any finding. It has been the rate at
which findings were withdrawn: six claims retracted, six explanations killed
by their own tests. Every one of those was caught by a control — a noise
floor, a random baseline, a dimension match, a second corpus — and every one
of those controls had to be *thought of* at the time, by someone who had just
finished being pleased with the result.

That is the weak link. The discipline lived in attention, and attention is
the part that does not persist. A claim made on a tired afternoon with no
random baseline looks exactly like a claim made carefully, once both are
prose in a README.

So the claims stop being prose. Each one is a record carrying its effect
size, the noise floor it was judged against, the controls that were actually
applied to it, and what became of it. `python -m sens claims --audit` then
asks the question I kept having to remember to ask:

    is this claim entitled to the confidence it is stated with?

and answers it mechanically, for all of them, every time — including for
claims I am no longer thinking about, which is the only kind that needs it.

The register is deliberately not clever. It is a list, and its value is that
it is exhaustive and that the tests refuse to let it fall out of step with
the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# the controls, and what each one rules out
# ---------------------------------------------------------------------------

CONTROLS: dict[str, str] = {
    "noise-floor": (
        "the effect was compared against the spread of identical rebuilds, "
        "so 'bigger' means bigger than chance and not merely bigger"
    ),
    "second-corpus": (
        "re-measured on a corpus sharing nothing but the language, which is "
        "what caught the claim this repository once led with"
    ),
    "paired-test": (
        "both systems answered the same questions and only the "
        "disagreements were counted"
    ),
    "dimension-matched": (
        "the compared spaces have the same number of axes, so the result is "
        "not measuring dimensionality"
    ),
    "random-baseline": (
        "compared against a random selection of the same size — the control "
        "whose absence made the sign ablation mean the opposite of what it "
        "was reported to mean"
    ),
    "shared-questions": (
        "the compared systems were scored on the intersection of what each "
        "could answer, not on whatever each happened to cover"
    ),
    "intervention": (
        "the proposed cause was manipulated directly rather than observed "
        "alongside the effect"
    ),
    "held-out": (
        "scored against text the model was not built on"
    ),
    "unbiased-metric": (
        "the metric has no structural reason to prefer the winner"
    ),
}

# A claim of this status is only entitled to it if these controls were run.
#
# The first version of this table asked only for a noise floor, and reported
# that every claim was entitled to its status. That answer was useless: the
# claim this repository most regrets — that compression trades identity for
# category — had a noise floor, three agreeing measurements, and one corpus.
# It passed. A register that cannot embarrass its author is decoration.
#
# So `holds` and `backwards` now demand replication. Saying a thing holds is
# a claim about the method; showing it on one corpus is a claim about one
# corpus, and this repository has already confused those once.
REQUIRED: dict[str, frozenset[str]] = {
    "holds": frozenset({"noise-floor", "second-corpus"}),
    # Solid where it was measured, and measured in one place. Distinguished
    # from `marginal` because conflating "weak effect" with "unreplicated
    # effect" hides which of the two a reader should discount.
    "single-corpus": frozenset({"noise-floor"}),
    "backwards": frozenset({"noise-floor", "second-corpus"}),
    "refuted": frozenset({"noise-floor", "second-corpus", "paired-test"}),
    "marginal": frozenset({"noise-floor"}),
    "no-effect": frozenset({"noise-floor"}),
    "open": frozenset(),
}

STATUSES = tuple(REQUIRED)

# The spread a held-out effect shows when nothing about the method changes —
# only the factorisation seed, the pair sample and the block size, all of
# which are choices nobody made deliberately. `sens robustness` measures it.
#
# Every sigma below is the effect divided by this. It used to be divided by
# the factorisation seed alone, at 0.0047, which overstated every claim in
# the repository by a factor of 2.4. The pair sample turned out to be the
# largest source and had never been looked at.
EFFECT_SD = 0.0111


@dataclass(frozen=True)
class ConfigDelta:
    """How to re-derive a claim: rebuild with one parameter changed.

    The register's numbers were all typed in by hand, which makes it a
    checklist that trusts its own entries. This is the part that does not:
    given a parameter and an alternative value, the effect can be recomputed
    from the corpus and compared against what was written down. A refactor
    that silently changes a result, or a digit transposed while writing the
    README, then stops being invisible.
    """

    parameter: str
    against: object

    @property
    def summary(self) -> str:
        return f"{self.parameter} vs {self.against!r}"


@dataclass(frozen=True)
class Claim:
    """One assertion, with the evidence it is entitled to lean on."""

    id: str
    what: str
    section: str
    status: str
    novels: str
    expository: str = "not run"
    sigma: float | None = None
    controls: frozenset[str] = field(default_factory=frozenset)
    note: str = ""
    # The machine-checkable part. `effect` is the number `novels` states in
    # prose; `check` says how to get it back out of the corpus.
    effect: float | None = None
    effect_expository: float | None = None
    unit: str = ""
    check: "ConfigDelta | None" = None

    @property
    def verifiable(self) -> bool:
        return self.check is not None and self.effect is not None

    def recorded(self, collection: str) -> float | None:
        """The effect this claim records for a given corpus."""
        if collection == "expository":
            return self.effect_expository
        return self.effect

    def checkable_on(self, collection: str) -> bool:
        return self.check is not None and self.recorded(collection) is not None

    @property
    def missing(self) -> frozenset[str]:
        """Controls this claim's status requires and does not have."""
        return REQUIRED[self.status] - self.controls

    @property
    def entitled(self) -> bool:
        return not self.missing

    @property
    def replicated(self) -> bool:
        return "second-corpus" in self.controls


def _c(*names: str) -> frozenset[str]:
    for name in names:
        if name not in CONTROLS:
            raise KeyError(f"unknown control {name!r}")
    return frozenset(names)


REGISTER: tuple[Claim, ...] = (
    # --- the pipeline stages, largest effect first ------------------------
    Claim(
        id="ppmi-vs-raw",
        what="PPMI beats using the raw co-occurrence counts",
        section="is-the-surprise-step-worth-anything",
        status="holds",
        novels="+0.3531 spearman",
        expository="+0.4787 spearman",
        sigma=31.8,
        controls=_c("noise-floor", "second-corpus", "held-out"),
        note="largest effect measured here; the step the README always "
             "claimed was the important one, finally with a control",
        effect=0.3531,
        effect_expository=0.4787,
        unit="spearman",
        check=ConfigDelta("weighting", 'raw'),
    ),
    Claim(
        id="ppmi-vs-log",
        what="PPMI beats log-compressed counts, so the value is the "
             "comparison against independence and not range squashing",
        section="is-the-surprise-step-worth-anything",
        status="holds",
        novels="+0.2669 spearman",
        expository="+0.2802 spearman",
        sigma=24.0,
        controls=_c("noise-floor", "second-corpus", "held-out"),
        effect=0.2669,
        effect_expository=0.2802,
        unit="spearman",
        check=ConfigDelta("weighting", 'log'),
    ),
    Claim(
        id="cosine-vs-dot",
        what="normalising beats the plain dot product",
        section="is-cosine-the-right-question-to-ask",
        status="holds",
        novels="+20.2 form points",
        expository="dot collapses to 3.6%",
        sigma=26.0,
        controls=_c("noise-floor", "second-corpus"),
        note="vector length correlates -0.77 with frequency rank, so the "
             "dot product largely reports which words are common",
    ),
    Claim(
        id="harmonic-window",
        what="weighting a neighbour by 1/distance beats a flat window",
        section="auditing-the-rest-of-the-defaults",
        status="holds",
        novels="+0.0858 spearman",
        expository="+0.1179 spearman",
        sigma=7.7,
        controls=_c("noise-floor", "second-corpus", "held-out"),
        note="the parameter that had nothing behind it but a sentence of "
             "prose turned out to be among the largest effects",
        effect=0.0858,
        effect_expository=0.1179,
        unit="spearman",
        check=ConfigDelta("harmonic", False),
    ),
    Claim(
        id="eigenvalue-exponent",
        what="scaling axes by |eigenvalue| beats the conventional square root",
        section="being-wrong-about-a-default",
        status="holds",
        novels="+0.1083 spearman",
        expository="+0.1283 spearman",
        sigma=9.8,
        controls=_c("noise-floor", "second-corpus", "held-out",
                    "unbiased-metric"),
        note="both curves turn over at an interior optimum, which is what "
             "rules out the metric merely rewarding reconstruction",
        effect=0.1083,
        effect_expository=0.1283,
        unit="spearman",
        check=ConfigDelta("eigenvalue_power", 0.5),
    ),
    Claim(
        id="clipping",
        what="clipping negative PMI beats keeping it",
        section="is-the-surprise-step-worth-anything",
        status="holds",
        novels="+0.0822 spearman",
        expository="+0.0837 spearman",
        sigma=7.4,
        controls=_c("noise-floor", "second-corpus", "held-out"),
        note="on held-out only; on analogy the two are indistinguishable",
        effect=0.0822,
        effect_expository=0.0837,
        unit="spearman",
        check=ConfigDelta("weighting", 'pmi'),
    ),
    Claim(
        id="pair-pruning",
        what="discarding pairs seen once or less beats keeping everything",
        section="auditing-the-rest-of-the-defaults",
        status="holds",
        novels="+0.0533 spearman",
        expository="+0.0832 spearman",
        sigma=4.8,
        controls=_c("noise-floor", "second-corpus", "held-out"),
        effect=0.0533,
        effect_expository=0.0832,
        unit="spearman",
        check=ConfigDelta("min_pair_weight", 0.0),
    ),
    Claim(
        id="no-shift",
        what="not shifting PMI beats shifting it",
        section="auditing-the-rest-of-the-defaults",
        status="holds",
        novels="+0.0426 spearman",
        expository="+0.0408 spearman",
        sigma=3.8,
        controls=_c("noise-floor", "second-corpus", "held-out"),
        effect=0.0426,
        effect_expository=0.0408,
        unit="spearman",
        check=ConfigDelta("shift", 2.0),
    ),
    Claim(
        id="cosine-vs-euclidean",
        what="cosine beats Euclidean distance",
        section="is-cosine-the-right-question-to-ask",
        status="holds",
        novels="+5.7 form points",
        expository="+4.0 form points, p=0.006",
        sigma=7.5,
        controls=_c("noise-floor", "second-corpus", "paired-test"),
        note="the first measurement of this was an artefact: the analogy "
             "target was built in unit space and compared in raw space",
    ),
    Claim(
        id="lowercasing",
        what="folding case beats keeping it",
        section="tokenisation-the-last-stage-and-two-defaults-that-lose",
        status="holds",
        novels="+0.0131 spearman, and 15% more vocabulary covered",
        expository="+0.0137 spearman, and 5.7% more vocabulary covered",
        sigma=3.4,
        controls=_c("noise-floor", "second-corpus", "shared-questions",
                    "held-out"),
        note="replicated when the register demanded it: same direction, "
             "same magnitude, 5.3 sd against the local floor",
    ),
    Claim(
        id="magnitude-ranking",
        what="ranking directions by |eigenvalue| beats ranking by sign",
        section="ranking-by-magnitude-earns-its-place-the-negatives-do-not",
        status="holds",
        novels="+2.4 top-1 points; 0 of 12 random draws reached it",
        expository="form 16.9% by magnitude, 13.5% random, 12.5% by sign",
        sigma=None,
        controls=_c("noise-floor", "second-corpus", "dimension-matched",
                    "random-baseline"),
        note="the effect is real; the original explanation for it was not. "
             "sign-based selection loses to random selection of the same "
             "size on both corpora, so sign is anti-correlated with "
             "importance rather than carrying signal of its own",
    ),

    # --- marginal ----------------------------------------------------------
    Claim(
        id="alpha-smoothing-off",
        what="no context-distribution smoothing beats the usual 0.75",
        section="auditing-the-rest-of-the-defaults",
        status="no-effect",
        novels="+0.0042 spearman",
        expository="same direction",
        sigma=0.4,
        controls=_c("noise-floor", "second-corpus", "held-out"),
        note="the default moved to 1.0 on a 2.4 sd result. Re-derived "
             "after the document-split fix it is 0.9 sd — within noise. The "
             "two settings are indistinguishable; 1.0 stays because it is "
             "the simpler of two equals, not because it is better",
        effect=0.0042,
        unit="spearman",
        check=ConfigDelta("alpha", 0.75),
    ),
    Claim(
        id="vocabulary-size",
        what="8000 words beats the default 4000",
        section="the-two-that-needed-a-different-ruler",
        status="marginal",
        novels="+0.0079 spearman",
        sigma=2.1,
        controls=_c("noise-floor", "shared-questions", "held-out"),
        note="not adopted: marginal gain, real cost, same reasoning as dim",
    ),
    Claim(
        id="window-size",
        what="a window of 4 beats a window of 2",
        section="auditing-the-rest-of-the-defaults",
        status="marginal",
        novels="+0.0219 spearman",
        sigma=2.0,
        controls=_c("noise-floor", "held-out"),
        note="recorded at 2.0 sd, then 4.0, then 4.6, now 2.0 again — "
             "twice moved by a stale baseline, then halved when the noise "
             "floor stopped counting only the factorisation seed",
        effect=0.0219,
        unit="spearman",
        check=ConfigDelta("window", 2),
    ),

    # --- measured and absent ------------------------------------------------
    Claim(
        id="window-4-vs-6",
        what="a window of 4 beats a window of 6",
        section="auditing-the-rest-of-the-defaults",
        status="no-effect",
        novels="-0.0039 spearman",
        sigma=0.4,
        controls=_c("noise-floor", "held-out"),
        note="anything between 4 and 6 is the same, and which of them is "
             "nominally ahead flips with the baseline — it flipped when the "
             "alpha default moved",
        effect=-0.0039,
        unit="spearman",
        check=ConfigDelta("window", 6),
    ),
    Claim(
        id="min-count",
        what="the minimum-occurrence cutoff matters",
        section="the-two-that-needed-a-different-ruler",
        status="no-effect",
        novels="0.0019 spearman across a fourfold range",
        sigma=0.5,
        controls=_c("noise-floor", "shared-questions", "held-out"),
        note="a knob connected to nothing, worth knowing because it looks "
             "like it should matter",
    ),
    Claim(
        id="3cosmul",
        what="multiplicative analogy beats vector-offset on a small corpus",
        section="morphological-analogy",
        status="no-effect",
        novels="-0.001 top-1",
        sigma=0.4,
        controls=_c("noise-floor"),
        note="predicted from the literature, did not appear",
    ),

    # --- true but backwards --------------------------------------------------
    Claim(
        id="accurate-factorisation",
        what="a more accurate factorisation makes the model worse",
        section="a-better-factorisation-that-made-a-worse-model",
        status="refuted",
        novels="-4.9 form points, 6 sd, while 39% faster and twice as "
               "accurate on eigenvalues",
        expository="+0.4 form points, p=0.87; top5 p=0.41. no difference",
        controls=_c("noise-floor", "second-corpus", "paired-test",
                    "unbiased-metric"),
        note="held as `backwards` until the register demanded replication, "
             "which it then failed. The regularisation story explained a "
             "real effect on one corpus and there is no effect to explain "
             "on the other. Krylov remains 39% faster and no worse there",
    ),

    # --- withdrawn -----------------------------------------------------------
    Claim(
        id="compression-trades-identity",
        what="compression buys category structure at the cost of identity",
        section="does-any-of-it-generalise",
        status="refuted",
        novels="+5.3 form points, p=0.004",
        expository="-5.3 form points, p=0.008",
        controls=_c("noise-floor", "second-corpus", "paired-test"),
        note="reached by three independent routes, all measuring the same "
             "six novels. The routes agreeing was not evidence",
    ),
    Claim(
        id="dimension-sweet-spot",
        what="64 dimensions beat 160 on category structure",
        section="how-many-dimensions-and-the-trade-seen-directly",
        status="refuted",
        novels="p=0.034 favouring 64",
        expository="p=0.91, no effect",
        controls=_c("noise-floor", "second-corpus", "paired-test"),
    ),

    # --- open ----------------------------------------------------------------
    Claim(
        id="past-tense-deficit",
        what="uncompressed PPMI rows encode past tense worse in narrative "
             "fiction than in expository prose, for the same verbs",
        section="where-the-difference-actually-lives",
        status="open",
        novels="raw produces an -ed form 28.3% of the time",
        expository="46.7% on the identical 120 questions, z=2.94, p=0.003",
        controls=_c("second-corpus", "shared-questions", "intervention"),
        note="not vocabulary (shared questions), not row density "
             "(pruning intervention moved it 0.0 points), not any gross "
             "corpus statistic. Unexplained, and left that way",
    ),
)


def by_status(status: str) -> list[Claim]:
    return [c for c in REGISTER if c.status == status]


def unentitled() -> list[Claim]:
    """Claims stated with more confidence than their controls support."""
    return [c for c in REGISTER if not c.entitled]


def unreplicated() -> list[Claim]:
    """Claims that have only ever been measured on one corpus."""
    return [c for c in REGISTER if not c.replicated]


def header() -> str:
    return f"{'id':<28} {'status':<10} {'sd':>6}  effect (novels)"


def row(claim: Claim) -> str:
    sigma = f"{claim.sigma:6.1f}" if claim.sigma is not None else "     -"
    return f"{claim.id:<28} {claim.status:<10} {sigma}  {claim.novels}"
