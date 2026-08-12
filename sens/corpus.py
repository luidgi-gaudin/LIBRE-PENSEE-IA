"""Where the words come from.

One book ships with the repository so that a clone with no network can still
build something. The rest are a download away. They are all out of copyright,
all long, and all English prose from a narrow enough period that the model
does not have to reconcile two centuries of usage at once.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"
)


@dataclass(frozen=True)
class Work:
    slug: str
    gutenberg_id: int
    title: str
    author: str
    bundled: bool = False

    @property
    def filename(self) -> str:
        return f"{self.slug}.txt"

    @property
    def path(self) -> str:
        return os.path.join(CORPUS_DIR, self.filename)

    @property
    def url(self) -> str:
        return (
            f"https://www.gutenberg.org/cache/epub/"
            f"{self.gutenberg_id}/pg{self.gutenberg_id}.txt"
        )


LIBRARY: tuple[Work, ...] = (
    Work("moby-dick", 2701, "Moby-Dick", "Melville", bundled=True),
    Work("frankenstein", 84, "Frankenstein", "Shelley"),
    Work("pride-and-prejudice", 1342, "Pride and Prejudice", "Austen"),
    Work("middlemarch", 145, "Middlemarch", "Eliot"),
    Work("war-and-peace", 2600, "War and Peace", "Tolstoy"),
    Work("monte-cristo", 1184, "The Count of Monte Cristo", "Dumas"),
)

# A second corpus, for asking whether any of this generalises.
#
# Every number in this repository was measured on six novels, which makes
# every conclusion a claim about six novels until something else is tried.
# These nine are chosen to be as unlike them as public-domain English gets
# while staying the same size: expository and argumentative prose rather
# than narrative, with almost no dialogue and a technical vocabulary.
EXPOSITORY: tuple[Work, ...] = (
    Work("origin-of-species", 1228, "On the Origin of Species", "Darwin"),
    Work("voyage-of-the-beagle", 944, "The Voyage of the Beagle", "Darwin"),
    Work("descent-of-man", 2300, "The Descent of Man", "Darwin"),
    Work("wealth-of-nations", 3300, "The Wealth of Nations", "Smith"),
    Work("leviathan", 3207, "Leviathan", "Hobbes"),
    Work("republic", 1497, "The Republic", "Plato"),
    Work("meditations", 2680, "Meditations", "Aurelius"),
    Work("beyond-good-and-evil", 4363, "Beyond Good and Evil", "Nietzsche"),
    Work("problems-of-philosophy", 5827, "The Problems of Philosophy", "Russell"),
)

COLLECTIONS: dict[str, tuple[Work, ...]] = {
    "novels": LIBRARY,
    "expository": EXPOSITORY,
}


def works(collection: str = "novels") -> tuple[Work, ...]:
    try:
        return COLLECTIONS[collection]
    except KeyError:
        raise KeyError(
            f"unknown collection {collection!r}; "
            f"expected one of {sorted(COLLECTIONS)}"
        ) from None


def available(collection: str = "novels") -> list[Work]:
    """The works currently present on disk."""
    return [w for w in works(collection) if os.path.exists(w.path)]


def missing(collection: str = "novels") -> list[Work]:
    return [w for w in works(collection) if not os.path.exists(w.path)]


def fetch(
    wanted: list[Work] | None = None,
    timeout: int = 60,
    collection: str = "novels",
) -> list[Work]:
    """Download any works not already on disk. Returns what was fetched."""
    os.makedirs(CORPUS_DIR, exist_ok=True)
    fetched = []
    for work in wanted if wanted is not None else works(collection):
        if os.path.exists(work.path):
            continue
        with urllib.request.urlopen(work.url, timeout=timeout) as response:
            data = response.read()
        # Write via a temporary name so an interrupted download never leaves
        # a truncated file that looks complete on the next run.
        temporary = work.path + ".part"
        with open(temporary, "wb") as handle:
            handle.write(data)
        os.replace(temporary, work.path)
        fetched.append(work)
    return fetched
