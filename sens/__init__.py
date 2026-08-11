"""sens — meaning from counting, in pure Python.

`sens` is French for both *meaning* and *direction*, which is the entire
thesis of this package compressed into four letters: the way you get the
first is by building the second.

    from sens.pipeline import build
    space, report = build(["corpus/moby-dick.txt"])
    space.neighbors("whale")
"""

from .counts import cooccurrence
from .pipeline import Config, build
from .space import Space
from .text import Vocabulary, tokenize
from .weight import ppmi

__all__ = [
    "Config",
    "Space",
    "Vocabulary",
    "build",
    "cooccurrence",
    "ppmi",
    "tokenize",
]

__version__ = "1.0.0"
