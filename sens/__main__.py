"""Command line for asking the space questions.

    python -m sens fetch
    python -m sens build
    python -m sens neighbors whale
    python -m sens analogy king man woman
    python -m sens similarity ship sea
    python -m sens axis sea land wind sailor house door
    python -m sens demo
    python -m sens info
"""

from __future__ import annotations

import argparse
import os
import sys

from . import corpus
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
    pending = corpus.missing()
    if not pending:
        print("corpus complete")
    for work in pending:
        print(f"fetching {work.title} ({work.author}) ...", flush=True)
        corpus.fetch([work])
    have = corpus.available()
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

    p = sub.add_parser("demo", help="a guided tour of the results")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("info", help="describe the built space")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
