"""From bytes to types.

Nothing here knows what a word means. It only knows how to find the edges
of one, and how to decide which ones are frequent enough to be worth a row
in the matrix.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

_GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)

# A word is letters, optionally stitched together by apostrophes: "whale",
# "whale's", "o'clock". Digits and punctuation are boundaries, not content.
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")


def strip_boilerplate(text: str) -> str:
    """Drop the Project Gutenberg header and footer, if present.

    The licence text is the same in every file, so leaving it in would teach
    the model that `gutenberg` and `ebook` are close relatives — true, but
    an artifact of packaging rather than of English.
    """
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end() :]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


# With case preserved the token pattern has to admit capitals too.
_TOKEN_CASED = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

# `whale's` -> `whale` + `'s`; `don't` -> `don` + `'t`. Splitting here is
# what decides whether the possessive is a word of its own or a suffix
# fused onto every noun that takes one.
_CLITIC = re.compile(r"^(.*?)('(?:s|t|ll|re|ve|d|m))$")


def normalise(
    text: str, lowercase: bool = True, fold_accents: bool = True
) -> str:
    """Fold typography down to something regular.

    Curly apostrophes become straight ones first, so `don’t` and `don't`
    are the same token rather than two strangers. That much is not optional;
    the other two are choices, and choices in this repository get measured.
    """
    text = text.replace("’", "'").replace("ʼ", "'")
    if fold_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower() if lowercase else text


def tokenize(
    text: str,
    lowercase: bool = True,
    fold_accents: bool = True,
    split_clitics: bool = False,
) -> Iterator[str]:
    """Yield word tokens in order of appearance.

    `split_clitics` decides whether `whale's` is one token or two. Keeping it
    whole is what gives this corpus a `possessive` benchmark category at all;
    splitting it turns every possessive into the same shared `'s` token, so
    the relation stops being a property of each word and becomes a word in
    its own right.
    """
    pattern = _TOKEN if lowercase else _TOKEN_CASED
    for match in pattern.finditer(
        normalise(text, lowercase=lowercase, fold_accents=fold_accents)
    ):
        token = match.group(0)
        if split_clitics:
            parts = _CLITIC.match(token)
            if parts and parts.group(1):
                yield parts.group(1)
                yield parts.group(2)
                continue
        yield token


def read_tokens(
    path: str,
    lowercase: bool = True,
    fold_accents: bool = True,
    split_clitics: bool = False,
) -> list[str]:
    """Read one corpus file into a token list."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return list(tokenize(
            strip_boilerplate(handle.read()),
            lowercase=lowercase,
            fold_accents=fold_accents,
            split_clitics=split_clitics,
        ))


@dataclass
class Vocabulary:
    """The set of words that get to exist, in descending frequency order.

    Index 0 is the most common word in the corpus. That ordering is not
    cosmetic: it means a vocabulary is a prefix of a larger vocabulary built
    from the same corpus, so cutting the size only ever removes the tail.
    """

    words: list[str]
    counts: list[int]

    def __post_init__(self) -> None:
        self.index: dict[str, int] = {w: i for i, w in enumerate(self.words)}

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    @property
    def total(self) -> int:
        return sum(self.counts)

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        max_size: int = 4000,
        min_count: int = 5,
    ) -> "Vocabulary":
        frequencies = Counter(tokens)
        # Sort by count, then alphabetically, so ties do not depend on the
        # iteration order of a dict and two identical corpora give identical
        # vocabularies.
        ordered = sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [(w, c) for w, c in ordered if c >= min_count][:max_size]
        return cls(words=[w for w, _ in kept], counts=[c for _, c in kept])

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Map tokens to ids, dropping anything out of vocabulary.

        Dropped words close the gap rather than leaving a hole, so a rare
        word between two common ones makes them neighbours. This is the
        conventional choice and it costs a little precision in the window
        distances to buy a lot of density in the counts.
        """
        lookup = self.index
        return [lookup[t] for t in tokens if t in lookup]
