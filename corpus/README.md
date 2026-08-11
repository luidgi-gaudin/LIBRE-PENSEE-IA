# Corpus

Six novels, 1.77 million words, all out of copyright, all from
[Project Gutenberg](https://www.gutenberg.org).

| file                      | work                      | author  | ~words |
| ------------------------- | ------------------------- | ------- | -----: |
| `moby-dick.txt`           | Moby-Dick                 | Melville | 216k |
| `frankenstein.txt`        | Frankenstein              | Shelley  |  78k |
| `pride-and-prejudice.txt` | Pride and Prejudice       | Austen   | 130k |
| `middlemarch.txt`         | Middlemarch               | Eliot    | 319k |
| `war-and-peace.txt`       | War and Peace             | Tolstoy  | 566k |
| `monte-cristo.txt`        | The Count of Monte Cristo | Dumas    | 464k |

Only `moby-dick.txt` is committed, so that a fresh clone can build something
without a network. The other five arrive with:

    python -m sens fetch

## Why these

They are long, which matters more than anything else — the method needs to
see a word in many different companies before its position settles.

They are close enough in period that the model is not asked to reconcile two
centuries of usage in one geometry, and far enough apart in subject that the
vocabulary is not all one thing. Whaling, a laboratory, a drawing room, a
provincial town, a war, a prison and a revenge: the shared words are the
general ones, which is exactly the vocabulary whose structure is interesting.

Two of them are translations, which is a real contaminant. Tolstoy and Dumas
reach the model through a translator's English, and translated prose has its
own distributional habits. They are in anyway, because 1.03 million words of
slightly odd English beats 743 thousand words of idiomatic English. Corpus
size wins ties.

## What this makes the model

Every result the model produces is a claim about *these six books*, not about
English. It is worth watching that boundary rather than forgetting it. Ask
this model for the neighbours of `king` and the top five are `xviii`,
`george`, `st`, `louis`, `king's` — not royalty in the abstract but Louis
XVIII, who appears in Dumas. Ask for `paris` and you get `rue`, `saint`,
`marseilles`, `de`, `meran`. The model has not learned about France; it has
learned Dumas.

That is not a defect to be corrected by cleaning the data. It is what a
distributional model *is*. It knows the company its words kept, and the
company was chosen here, in this table.
