"""Command line for asking the space questions.

    python -m sens fetch
    python -m sens build
    python -m sens neighbors whale
    python -m sens analogy king man woman
    python -m sens similarity ship sea
    python -m sens axis sea land wind sailor house door
    python -m sens evaluate
    python -m sens sweep
    python -m sens heldout
    python -m sens baseline
    python -m sens audit
    python -m sens dimensions
    python -m sens subspaces
    python -m sens noise
    python -m sens claims
    python -m sens demo
    python -m sens info
"""

from __future__ import annotations

import argparse
import os
import sys

from . import corpus
from . import evaluate as evaluate_mod
from . import experiments
from . import heldout as heldout_mod
from . import baseline as baseline_mod
from . import audit as audit_mod
from . import noise as noise_mod
from . import claims as claims_mod
from .pipeline import Config, build
from .space import Space

DEFAULT_SPACE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build",
    "space.sens",
)


def _load(path: str) -> Space:
    if not os.path.exists(path):
        sys.exit(
            f"no space at {path}\nrun `python -m sens build` first "
            f"(and `python -m sens fetch` before that for the full corpus)"
        )
    return Space.load(path)


def _table(rows: list[tuple[str, float]]) -> str:
    if not rows:
        return "  (nothing)"
    width = max(len(w) for w, _ in rows)
    return "\n".join(f"  {w:<{width}}  {s:+.4f}" for w, s in rows)


def cmd_fetch(args: argparse.Namespace) -> int:
    pending = corpus.missing(args.collection)
    if not pending:
        print("corpus complete")
    for work in pending:
        print(f"fetching {work.title} ({work.author}) ...", flush=True)
        corpus.fetch([work])
    have = corpus.available(args.collection)
    print(f"\n{len(have)} work(s) in {corpus.CORPUS_DIR}")
    for work in have:
        size = os.path.getsize(work.path) / 1e6
        print(f"  {work.title:<28} {work.author:<10} {size:5.1f} MB")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    works = corpus.available()
    if not works:
        sys.exit("no corpus files found; run `python -m sens fetch`")

    config = Config(
        vocab_size=args.vocab,
        window=args.window,
        dim=args.dim,
        min_pair_weight=args.min_pair_weight,
        power_iterations=args.power_iterations,
    )

    print(f"building from {len(works)} work(s)", file=sys.stderr)
    space, _ = build([w.path for w in works], config)

    os.makedirs(os.path.dirname(args.space), exist_ok=True)
    space.save(args.space)
    size = os.path.getsize(args.space) / 1e6
    print(
        f"\n{len(space)} words x {space.dim} dims -> {args.space} "
        f"({size:.1f} MB)",
        file=sys.stderr,
    )
    return 0


def cmd_neighbors(args: argparse.Namespace) -> int:
    space = _load(args.space)
    for word in args.words:
        if word not in space:
            print(f"{word}: not in vocabulary")
            continue
        print(f"{word}")
        print(_table(space.neighbors(word, n=args.n)))
        print()
    return 0


def cmd_analogy(args: argparse.Namespace) -> int:
    space = _load(args.space)
    missing = [w for w in (args.a, args.b, args.c) if w not in space]
    if missing:
        sys.exit(f"not in vocabulary: {', '.join(missing)}")
    print(f"{args.a} : {args.b} :: {args.c} : ?")
    print(_table(space.analogy(args.a, args.b, args.c, n=args.n)))
    return 0


def cmd_similarity(args: argparse.Namespace) -> int:
    space = _load(args.space)
    missing = [w for w in args.words if w not in space]
    if missing:
        sys.exit(f"not in vocabulary: {', '.join(missing)}")
    for i, a in enumerate(args.words):
        for b in args.words[i + 1 :]:
            print(f"  {a} ~ {b}: {space.similarity(a, b):+.4f}")
    return 0


def cmd_axis(args: argparse.Namespace) -> int:
    space = _load(args.space)
    for end in (args.negative, args.positive):
        if end not in space:
            sys.exit(f"not in vocabulary: {end}")
    words = args.words or space.words[: args.n]
    print(f"{args.negative}  <------------------->  {args.positive}")
    print(_table(space.axis(args.negative, args.positive, words)))
    return 0


DEMO_NEIGHBORS = ["whale", "captain", "sea", "happiness", "paris"]
DEMO_ANALOGIES = [
    ("he", "his", "she"),
    ("father", "mother", "son"),
    ("king", "queen", "man"),
]
DEMO_AXIS = (
    "sea",
    "land",
    "ship whale sailor harpoon mast deck "
    "house door garden village horse field forest road".split(),
)


def cmd_demo(args: argparse.Namespace) -> int:
    """A guided tour, so the repository can show its own results."""
    space = _load(args.space)

    print("=" * 62)
    print("NEIGHBOURS — what the geometry puts next to what")
    print("=" * 62)
    for word in DEMO_NEIGHBORS:
        if word in space:
            print(f"\n{word}")
            print(_table(space.neighbors(word, n=8)))

    print("\n" + "=" * 62)
    print("ANALOGY — the same offset, applied somewhere else")
    print("=" * 62)
    for a, b, c in DEMO_ANALOGIES:
        if all(w in space for w in (a, b, c)):
            print(f"\n{a} : {b} :: {c} : ?")
            print(_table(space.analogy(a, b, c, n=3)))

    print("\n" + "=" * 62)
    print("AXIS — a contrast nobody wrote down, recovered as a direction")
    print("=" * 62)
    negative, positive, words = DEMO_AXIS
    if negative in space and positive in space:
        print(f"\n{negative}  <------------------->  {positive}")
        print(_table(space.axis(negative, positive, words)))
    print()
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    space = _load(args.space)
    categories = evaluate_mod.build_benchmark(space)
    if not categories:
        sys.exit("vocabulary too small to generate a benchmark")

    print("benchmark generated from the vocabulary")
    for category in categories:
        sample = ", ".join(f"{a}/{b}" for a, b in sorted(category.pairs)[:3])
        print(f"  {category.name:<12} {len(category.pairs):4} pairs   {sample}")

    scores, misses = evaluate_mod.evaluate(
        space, limit=args.limit, seed=args.seed
    )

    methods = sorted({s.method for s in scores})
    names = [c.name for c in categories]
    width = max(len(n) for n in names + ["ALL"])

    header = "  top1    top5    form"
    print(f"\n{'':<{width}}  " + "  ".join(f"{m:<24}" for m in methods))
    print(f"{'':<{width}}  " + "  ".join(f"{header:<24}" for _ in methods))

    def row(label, pick):
        cells = []
        for method in methods:
            hit = pick(method)
            cells.append(
                f"{hit.accuracy:6.1%}  {hit.recall5:6.1%}  {hit.form_rate:6.1%}"
            )
        print(f"{label:<{width}}  " + "  ".join(cells))

    for name in names:
        row(name, lambda m, n=name: next(
            s for s in scores if s.category == n and s.method == m
        ))

    combined = evaluate_mod.totals(scores)
    row("ALL", lambda m: combined[m])
    print(f"\n{combined[methods[0]].asked} questions per method")

    if args.misses:
        print("\nwhat it says instead")
        for key in sorted(misses):
            examples = misses[key][: args.misses]
            if not examples:
                continue
            print(f"\n  {key}")
            for a, b, c, d, got in examples:
                print(f"    {a} : {b} :: {c} : {got}   (wanted {d})")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    space = _load(args.space)

    print("eigenvalue exponent")
    print("  " + experiments.header())
    for result in experiments.sweep_power(
        space, method=args.method, limit=args.limit, seed=args.seed
    ):
        print("  " + result.row)

    ablation = experiments.ablate_signs(
        space, method=args.method, limit=args.limit, seed=args.seed
    )
    if ablation:
        print("\nsign ablation (dimension-matched)")
        print("  " + experiments.header())
        for result in ablation:
            print("  " + result.row)
    else:
        print("\nno negative eigenvalues retained; nothing to ablate")
    return 0


DEFAULT_HELDOUT_POWERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def cmd_heldout(args: argparse.Namespace) -> int:
    works = corpus.available()
    if not works:
        sys.exit("no corpus files found; run `python -m sens fetch`")

    print("splitting the corpus and building on half of it", file=sys.stderr)
    prepared = heldout_mod.prepare(
        [w.path for w in works],
        block=args.block,
        pair_count=args.pairs,
        seed=args.seed,
        verbose=True,
    )
    print(
        f"\ntrain {prepared.train_tokens:,} tokens   "
        f"test {prepared.test_tokens:,} tokens   "
        f"{len(prepared.pairs):,} word pairs"
    )

    print("\nspearman against similarity measured on unseen text")
    print(f"  {'power':>6}  {'rho':>8}  {'pairs':>6}")
    best = None
    for power, rho, used in heldout_mod.sweep_power(
        prepared, tuple(args.powers)
    ):
        marker = ""
        if best is None or rho > best[1]:
            best = (power, rho)
        print(f"  {power:6.2f}  {rho:+8.4f}  {used:6}{marker}")
    if best:
        print(f"\nbest at power {best[0]:.2f} (rho {best[1]:+.4f})")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Compare the compressed space against not compressing at all."""
    from .counts import cooccurrence
    from .evaluate import build_benchmark
    from .heldout import split_blocks
    from .pipeline import Config
    from .text import read_tokens
    from .weight import ppmi

    works = corpus.available()
    if not works:
        sys.exit("no corpus files found; run `python -m sens fetch`")
    paths = [w.path for w in works]
    config = Config()

    print("building both representations on the same half-corpus",
          file=sys.stderr)
    prepared = heldout_mod.prepare(
        paths, config=config, seed=args.seed, verbose=True
    )

    tokens = []
    for path in paths:
        tokens.extend(read_tokens(path))
    train, _ = split_blocks(tokens)
    raw = ppmi(
        cooccurrence(
            prepared.vocab.encode(train),
            size=len(prepared.vocab),
            window=config.window,
            harmonic=config.harmonic,
            min_weight=config.min_pair_weight,
        ),
        alpha=config.alpha,
        shift=config.shift,
    )
    sparse = baseline_mod.SparseSpace(list(prepared.space.words), raw)

    result = baseline_mod.compare(
        prepared.space,
        sparse,
        build_benchmark(prepared.space),
        limit=args.limit,
        seed=args.seed,
    )

    print(f"\n{result.dense.asked} paired questions\n")
    print("  " + baseline_mod.header())
    print("  " + result.sparse.row)
    print("  " + result.dense.row)

    print("\npaired significance (McNemar, disagreements only)")
    for metric in ("top1", "top5", "form"):
        sparse_only, dense_only, p = result.significance(metric)
        winner = "raw" if sparse_only > dense_only else "compressed"
        verdict = f"{winner} wins" if p < 0.05 else "no difference"
        print(
            f"  {metric:<5} raw-only={sparse_only:<4} "
            f"compressed-only={dense_only:<4} p={p:.4f}   {verdict}"
        )
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    works = corpus.available()
    if not works:
        sys.exit("no corpus files found; run `python -m sens fetch`")

    print("building the ruler (fixed vocabulary, truth table and pairs)",
          file=sys.stderr)
    ruler = audit_mod.build_ruler(
        [w.path for w in works], seed=args.seed
    )
    print(
        f"vocab {len(ruler.vocab):,}   train {len(ruler.train):,}   "
        f"test {len(ruler.test):,}   pairs {len(ruler.pairs):,}\n"
    )
    print("this rebuilds the space once per value; expect several minutes\n",
          file=sys.stderr)

    def progress(parameter, value):
        print(f"  building {parameter}={value} ...", file=sys.stderr, flush=True)

    findings = audit_mod.audit(ruler, progress=progress)
    print("held-out spearman, one parameter varied at a time")
    for finding in findings:
        print("  " + finding.row)

    if args.vocabulary:
        print("\nvocabulary parameters, against a truth table from the "
              "largest vocabulary")
        vocab_findings = audit_mod.audit_vocabulary(
            [w.path for w in works], seed=args.seed, progress=progress
        )
        for finding in vocab_findings:
            print("  " + finding.row)
        findings = findings + vocab_findings

    wrong = [f for f in findings if not f.default_wins]
    print()
    if wrong:
        for f in wrong:
            print(f"  {f.parameter}: default {f.default} loses to {f.best[0]}")
    else:
        print("  every default is the best value on this grid")
    return 0


def cmd_dimensions(args: argparse.Namespace) -> int:
    """How many dimensions, judged by two metrics that disagree."""
    from .pipeline import Config

    works = corpus.available()
    if not works:
        sys.exit("no corpus files found; run `python -m sens fetch`")

    dims = tuple(sorted(args.dims))
    print(f"building once at {dims[-1]} dims; every smaller one is a prefix",
          file=sys.stderr)
    prepared = heldout_mod.prepare(
        [w.path for w in works],
        config=Config(dim=dims[-1]),
        seed=args.seed,
        verbose=True,
    )

    rows = experiments.dimension_curve(
        prepared, dims=dims, limit=args.limit, seed=args.seed
    )
    print()
    print("  " + experiments.dimension_header())
    for row in rows:
        print("  " + row.row)

    best_rho = max(rows, key=lambda r: r.heldout)
    best_form = max(rows, key=lambda r: r.form)
    print(
        f"\n  held-out similarity peaks at {best_rho.dims} dims; "
        f"form peaks at {best_form.dims}"
    )
    if best_rho.dims != best_form.dims:
        print("  the two metrics disagree, which is the finding, not a fault")
    return 0


def cmd_subspaces(args: argparse.Namespace) -> int:
    """Do the two factorisations disagree about scale, or about direction?"""
    from .pipeline import Config, build

    works = corpus.available()
    if not works:
        sys.exit("no corpus files found; run `python -m sens fetch`")
    paths = [w.path for w in works]

    print("building both factorisations of the same matrix", file=sys.stderr)
    left, _ = build(paths, Config(factoriser="subspace"), verbose=False)
    right, _ = build(paths, Config(factoriser="krylov"), verbose=False)

    print("\nprincipal angles, subspace iteration against block Krylov")
    print("cosine 1.0 = the two spaces share that direction exactly\n")
    print("  " + experiments.overlap_header())
    for overlap in experiments.compare_subspaces(left, right):
        print("  " + overlap.row)
    return 0


def cmd_noise(args: argparse.Namespace) -> int:
    """How big does a difference have to be before it means anything?"""
    works = corpus.available()
    if not works:
        sys.exit("no corpus files found; run `python -m sens fetch`")
    paths = [w.path for w in works]

    print("rebuilding under several seeds; expect several minutes",
          file=sys.stderr)
    ruler = audit_mod.build_ruler(paths)
    spreads = noise_mod.measure(
        paths, ruler, seeds=tuple(args.seeds), limit=args.limit,
        progress=lambda s: print(f"  seed {s} ...", file=sys.stderr, flush=True),
    )
    print(f"\n{len(args.seeds)} builds, identical but for the random seed\n")
    for spread in spreads:
        print("  " + spread.row)
    print(
        "\n  Differences smaller than the noise floor mean nothing. This is a"
        "\n  lower bound: the seed is the only thing varied here, so the real"
        "\n  uncertainty on any number is larger, never smaller."
    )
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    """Every claim this repository makes, and what it is entitled to."""
    register = claims_mod.REGISTER
    if args.audit:
        weak = claims_mod.unentitled()
        thin = claims_mod.unreplicated()
        print("claims stated with more confidence than their controls support")
        if weak:
            for claim in weak:
                missing = ", ".join(sorted(claim.missing))
                print(f"  {claim.id:<28} {claim.status:<10} missing: {missing}")
        else:
            print("  none — every status has the controls it requires")
        print(f"\nclaims measured on only one corpus ({len(thin)} of "
              f"{len(register)})")
        for claim in thin:
            print(f"  {claim.id:<28} {claim.status}")
        print("\ncontrols and what each rules out")
        for name in sorted(claims_mod.CONTROLS):
            used = sum(1 for c in register if name in c.controls)
            print(f"  {name:<18} {used:2}/{len(register)}  "
                  f"{claims_mod.CONTROLS[name]}")
        return 0

    for status in claims_mod.STATUSES:
        group = claims_mod.by_status(status)
        if not group:
            continue
        print(f"\n{status.upper()}  ({len(group)})")
        print("  " + claims_mod.header())
        for claim in group:
            print("  " + claims_mod.row(claim))
    replicated = sum(1 for c in register if c.replicated)
    print(f"\n{len(register)} claims, {replicated} checked on a second corpus")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    space = _load(args.space)
    meta = space.meta
    print(f"words        {len(space):,}")
    print(f"dimensions   {space.dim}")
    print(f"tokens       {meta.get('tokens', 0):,}")
    print("sources")
    for source in meta.get("sources", []):
        print(f"  {os.path.basename(source)}")
    print("config")
    for key, value in sorted(meta.get("config", {}).items()):
        print(f"  {key:<18} {value}")
    values = meta.get("eigenvalues", [])
    if values:
        head = ", ".join(f"{v:.1f}" for v in values[:6])
        print(f"eigenvalues  {head}, ... {values[-1]:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--space", default=DEFAULT_SPACE, help="space file")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the corpus")
    p.add_argument("--collection", default="novels",
                   choices=sorted(corpus.COLLECTIONS),
                   help="novels (default) or the expository control corpus")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", help="build a space from the corpus")
    p.add_argument("--vocab", type=int, default=4000)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--min-pair-weight", type=float, default=1.0)
    p.add_argument("--power-iterations", type=int, default=3)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("neighbors", help="nearest words")
    p.add_argument("words", nargs="+")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(func=cmd_neighbors)

    p = sub.add_parser("analogy", help="a is to b as c is to ?")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("c")
    p.add_argument("-n", type=int, default=5)
    p.set_defaults(func=cmd_analogy)

    p = sub.add_parser("similarity", help="pairwise cosine")
    p.add_argument("words", nargs="+")
    p.set_defaults(func=cmd_similarity)

    p = sub.add_parser("axis", help="project words onto a contrast")
    p.add_argument("negative")
    p.add_argument("positive")
    p.add_argument("words", nargs="*")
    p.add_argument("-n", type=int, default=20)
    p.set_defaults(func=cmd_axis)

    p = sub.add_parser("evaluate", help="score the space on a generated benchmark")
    p.add_argument("--limit", type=int, default=120,
                   help="questions per category per method")
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--misses", type=int, default=0,
                   help="show this many wrong answers per category")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sweep", help="test the pipeline's free parameters")
    p.add_argument("--method", default="3cosadd",
                   choices=["3cosadd", "3cosmul"])
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260811)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser(
        "heldout", help="predict similarity on text the space never saw"
    )
    p.add_argument("--block", type=int, default=4000)
    p.add_argument("--pairs", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--powers", type=float, nargs="+",
                   default=list(DEFAULT_HELDOUT_POWERS))
    p.set_defaults(func=cmd_heldout)

    p = sub.add_parser(
        "baseline", help="compare against not compressing at all"
    )
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260811)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser(
        "audit", help="check every default against a fixed ruler (slow)"
    )
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--vocabulary", action="store_true",
                   help="also audit vocab_size and min_count (slower)")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser(
        "dimensions", help="how many dimensions, on two metrics (slow)"
    )
    p.add_argument("--dims", type=int, nargs="+",
                   default=[16, 32, 48, 64, 96, 128, 160])
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260811)
    p.set_defaults(func=cmd_dimensions)

    p = sub.add_parser(
        "subspaces", help="do two factorisations find the same directions?"
    )
    p.set_defaults(func=cmd_subspaces)

    p = sub.add_parser(
        "noise", help="measure the noise floor of every metric (slow)"
    )
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(noise_mod.DEFAULT_SEEDS))
    p.add_argument("--limit", type=int, default=120)
    p.set_defaults(func=cmd_noise)

    p = sub.add_parser(
        "claims", help="every claim made here, and what supports it"
    )
    p.add_argument("--audit", action="store_true",
                   help="report claims whose controls do not match their status")
    p.set_defaults(func=cmd_claims)

    p = sub.add_parser("demo", help="a guided tour of the results")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("info", help="describe the built space")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
