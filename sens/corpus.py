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


def available() -> list[Work]:
    """The works currently present on disk."""
    return [w for w in LIBRARY if os.path.exists(w.path)]


def missing() -> list[Work]:
    return [w for w in LIBRARY if not os.path.exists(w.path)]


def fetch(works: list[Work] | None = None, timeout: int = 60) -> list[Work]:
    """Download any works not already on disk. Returns what was fetched."""
    os.makedirs(CORPUS_DIR, exist_ok=True)
    fetched = []
    for work in works if works is not None else LIBRARY:
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
