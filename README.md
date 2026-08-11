# sens

**Meaning from counting.** Six novels go in. What comes out is a space where
`whale` sits next to `sperm` and `fishery`, where `she` minus `he` plus `his`
lands on `her`, and where the line from `sea` to `land` sorts `harpoon` from
`garden` — none of which anybody wrote down anywhere.

Pure Python. No NumPy, no SciPy, no dependencies of any kind. Every
multiply-add is a Python float operation you could step through in a
debugger. 525 lines implement the method; the rest is a command line, a
corpus fetcher, an evaluation harness and the experiments that test the
pipeline's own settings. 174 tests. 42 seconds end to end.

```
$ python -m sens fetch && python -m sens build
  read           1,771,272 tokens from 6 file(s)       2.3s
  vocabulary     4,000 types, 90.5% of tokens kept     0.4s
  co-occurrence  742,430 pairs, 4.6% dense             3.9s
  ppmi           522,603 positive, 70.4% of pairs      0.2s
  factorise      64 dims, eigenvalue spread   26.6x   35.6s

$ python -m sens demo
```

## What it produces

Everything below is verbatim output from `python -m sens demo` on the
committed defaults. Nothing is cherry-picked from a better run; there is only
one run, because the seed is fixed.

```
whale                    captain                  sea
  sperm      +0.8582       ahab       +0.7338       clouds   +0.6798
  greenland  +0.8160       mate       +0.7139       ocean    +0.6742
  ship       +0.7123       starbuck   +0.7086       sky      +0.6735
  bone       +0.7035       stubb      +0.7008       beneath  +0.6289
  fishery    +0.6905       peleg      +0.6856       wind     +0.6229
  whales     +0.6687       ship       +0.6755       rolling  +0.6187
  fish       +0.6432       steelkilt  +0.6627       water    +0.6160
  whale's    +0.6405       mates      +0.6419       ship's   +0.6116
```

It has never been told that Ahab and Starbuck are both people on a ship. It
has counted.

The clearest result is not the neighbour lists but the axis. Take the
direction from `sea` to `land` and project unrelated words onto it:

```
sea  <------------------->  land
  ship     -0.3690
  whale    -0.3496
  deck     -0.2684
  mast     -0.1790
  sailor   -0.1603
  harpoon  -0.0946
  ------------------ 0
  horse    +0.0232
  garden   +0.0511
  forest   +0.1325
  door     +0.1343
  road     +0.1696
  field    +0.1767
  village  +0.2490
  house    +0.2802
```

Fourteen words, no errors, and the zero crossing falls exactly where a person
would put it. That contrast appears in no dictionary the program has read. It
is a *direction*, recovered from nothing but a table of who stood near whom.

## How it works

Five stages. Three are bookkeeping, one makes a claim, and one produces the
meaning.

**1. Tokens.** Lowercase, strip accents, split on anything that is not a
letter or an internal apostrophe. Keep the 4,000 most frequent types; that
covers 90.5% of the corpus, because word frequencies are Zipfian and the
tail is enormous and useless.

**2. Counts.** For every word, tally what appeared within four places of it,
weighting a neighbour by `1/distance`. That decay is a cheap stand-in for the
fact that syntactic relations are mostly local. Without it the far edge of
the window contributes as much evidence as the word right next door, and the
whole matrix blurs.

**3. Surprise.** This is the only step that makes a claim about language.

A raw count says almost nothing. `the` sits next to everything, so a large
count with `the` is evidence that `the` is common, not that a relationship
exists. What carries information is the *gap* between how often two words
co-occur and how often they would if they were independent:

```
pmi(i, j) = log( p(i, j) / ( p(i) · p(j) ) )
```

Negatives are clipped to zero. That is not tidiness. A negative score asserts
that two words *avoid* each other, and at this corpus size almost every such
assertion is noise about a pair that simply never had the chance to meet.
Clipping throws away a real signal to avoid a much larger imaginary one.

**4. Factorisation.** The 4,000 × 4,000 matrix of surprise gets compressed to
4,000 × 64 by truncated SVD — implemented here as a randomised range finder
plus Jacobi rotations, both by hand, in `linalg.py`.

**This is the step that creates meaning, and it does so by destroying
information.** Before it, every word is a 4,000-long list of the specific
words it happened to appear near — a record, precise and useless. `sea` and
`ocean` share almost none of those entries, because a sentence that needs one
does not then need the other. Force all 4,000 words to share 64 axes and
precision becomes unaffordable. The only way to fit is to spend dimensions on
regularities that pay off across many words at once, and *being the sort of
thing sailors are near* is such a regularity while *appearing in line 41,822*
is not. Generalisation is what a lossy encoder does when it runs out of room.

**5. Geometry.** Scale each axis by `|eigenvalue|^0.5`, normalise, and ask
questions with dot products.

One detail worth keeping: the fourth-largest eigenvalue of this matrix is
**−98.8**. A PPMI matrix is symmetric but *indefinite*, so its strongest
directions are not all positive ones, and a factorisation that ranked by
eigenvalue instead of by magnitude would throw that axis away. Ranking by
`|λ|` is what makes this a truncated SVD rather than a truncated
eigendecomposition, and the difference is not cosmetic.

## Measuring it

Neighbour lists are easy to admire and impossible to argue with. You read
`whale -> sperm, fishery, whales`, you nod, and you have learned nothing you
could compare against another build. So `python -m sens evaluate` generates a
benchmark **from the vocabulary itself** — no download, no human judgement.
English morphology is regular enough that `walk : walked :: work : worked`
can be constructed by string manipulation, and the vocabulary is its own
filter: a bad stem yields a non-word, and non-words are not in the top 4,000.

```
             3cosadd                   3cosmul
               top1    top5    form      top1    top5    form
plural         7.5%   17.5%   34.2%    6.7%   14.2%   30.8%
past           0.0%   10.0%   18.3%    0.0%    7.5%   22.5%
progressive    0.8%    5.0%   10.0%    0.8%    3.3%   10.0%
adverb         0.8%    0.8%   11.7%    0.0%    1.7%    9.2%
possessive    10.8%   31.7%   44.2%   10.8%   34.2%   46.7%
er-form        1.7%    5.0%    1.7%    1.7%    3.3%    1.7%
negation       3.3%    9.2%    6.7%    0.8%    8.3%    4.2%
ALL            3.6%   11.3%   18.1%    3.0%   10.4%   17.9%

840 questions per method
```

Two things came out of this that I did not expect.

**The multiplicative method didn't help.** I added 3CosMul because Levy and
Goldberg report it beating vector-offset, most of all on small corpora — this
corpus being about as small as they come. It came out *slightly worse*: 3.0%
against 3.6%. At this accuracy both are close enough to the floor that the
difference is not worth defending in either direction, which is itself the
finding. The prediction was clean, the measurement disagreed, and both are in
the repository.

**The `form` column is where the real result is.** It counts answers that are
the right *kind* of word even when they are the wrong word. Asked for
`bingley's`, the model says `darcy's`:

```
friend : friend's :: bingley : darcy's     (wanted bingley's)
count  : count's  :: dolokhov : prince's   (wanted dolokhov's)
```

That is not noise. On possessives the model produces *a possessive* 44% of
the time and *the right* possessive 11% of the time — a fourfold gap. The
relation is in the geometry; what fails is holding onto `c` while applying
it. Accuracy alone cannot tell `darcy's` from `darcy`, and those two failures
mean opposite things: one says the relation was never learned, the other says
it was learned and the identity leaked. Only the second is true here.

The `er-form` row is the control that makes the rest credible. Its form rate
is 1.7% — the floor — because `-er` is two relations wearing one suffix
(`bank`/`banker` is an agent, `hard`/`harder` a comparative) and the model
cannot encode a direction that points two ways at once. A metric that scored
everything highly would be measuring itself.

## Testing the pipeline's own choices

Two settings had been made by citation rather than measurement. Now that a
benchmark exists, `python -m sens sweep` checks them. Both experiments reuse
one factorisation — the exponent is applied after the eigendecomposition and
an ablation only drops columns — so each setting costs one evaluation rather
than one rebuild.

**The eigenvalue exponent was wrong, and I have left it wrong on purpose.**

```
configuration          dims    top1    top5    form
power 0.00               64    3.1%    7.6%   16.7%
power 0.25               64    3.8%   10.2%   16.2%
power 0.50               64    4.3%   11.2%   17.9%     <- the default
power 0.75               64    4.8%   13.1%   18.1%
power 1.00               64    4.8%   14.0%   22.9%
```

Accuracy climbs all the way to 1.0. It is not a seed artifact — across three
seeds, 0.0 is always worst and 0.5 is never the winner. Neighbour lists and
the `sea`/`land` axis are unchanged at 1.0, so nothing visibly breaks.

The default is still 0.5. Not out of caution: a higher exponent emphasises
the dominant directions, which is exactly the change you would expect to
flatter a global linear structure like analogy while costing local structure
like similarity. I can measure the first and not the second. Tuning a default
against the only metric I happen to own, in the direction that metric is
biased toward, would be overfitting with extra steps. Moving it needs a
second measurement, which is the next thing to build.

**The negative eigenvalues earn their place.** This one confirmed the design
rather than upsetting it, and it is dimension-matched so that it means
something — comparing a 49-axis space against a 64-axis one would measure
dimensionality, not sign.

```
configuration          dims    top1    top5    form
top 49 by |lambda|       49    4.3%   12.4%   16.9%
positive only (49)       49    1.9%    7.4%    7.6%
negative only (15)       15    0.5%    1.4%   12.9%
all 64                   64    4.3%   11.2%   17.9%
```

Same number of axes, more than double the accuracy, purely from letting 15
negative-eigenvalue directions in. Ranking by `|λ|` was justified earlier in
this README on the theoretical grounds that it is what makes a truncated SVD
a truncated SVD; it turns out to be worth a factor of two in practice.

The last row is the strangest. Those 15 negative directions, alone, score
almost nothing on accuracy — 0.5% — while reaching a 12.9% form rate, higher
than all 49 positive directions together manage. Whatever they encode is
closer to *what kind of word this is* than to *which word this is*.

## What doesn't work

Analogies are the famous demo and they are the weakest thing here.

```
he : his :: she : ?              father : mother :: son : ?
  her       +0.8943               brother   +0.7629
  mother's  +0.7075               sister    +0.7260
  wife's    +0.6999               daughter  +0.6643
```

The first is right. The second is *almost* right — `sister` and `daughter`
are both defensible, but `brother` wins, and `brother` is wrong. And the one
everybody quotes:

```
man : woman :: king : ?
  xviii   +0.5386
  george  +0.5178
  henry   +0.5046
```

That is not a near miss, it is a different question being answered. This
corpus's `king` is Louis XVIII from Dumas, not an abstract monarch, so the
model returns his regnal number. It is behaving correctly on the evidence it
was given. `king − man + woman = queen` needs billions of tokens, and this
has 1.8 million. Anyone reporting it working at this scale tuned until it did.

The other honest failure is more interesting. Project gendered words onto the
`he → she` axis and `husband` lands firmly on the `she` side, between `nurse`
and `wife`:

```
he  <------------------->  she
  captain    -0.2216        lady       +0.3150
  officer    -0.1915        husband    +0.3243
  gentleman  -0.1353        nurse      +0.3299
  soldier    -0.1112        wife       +0.3502
  sailor     -0.0981        daughter   +0.4316
```

Nothing has gone wrong. The axis was never a gender axis. It measures *who
gets talked about near the word `she`*, and in six 19th-century novels the
answer includes *her husband*. The method has no access to gender; it has
access to co-occurrence, and it reported co-occurrence accurately. The error
is in reading the axis as something it never claimed to be — which is worth
remembering every time a direction in a learned space gets given a name.

## Try it

Python 3.9+. Nothing to install.

```bash
git clone <this repo> && cd LIBRE-PENSEE-IA

python -m sens build                    # Moby-Dick only, offline, ~16s
python -m sens fetch                    # the other five novels
python -m sens build                    # all six, ~42s

python -m sens demo
python -m sens evaluate                 # score it on a generated benchmark
python -m sens evaluate --misses 5      # ...and see what it says instead
python -m sens sweep                    # test the pipeline's free parameters
python -m sens neighbors whale ship happiness
python -m sens analogy father mother son
python -m sens similarity ship boat garden
python -m sens axis sea land harpoon garden mast door
python -m sens info
```

Knobs worth turning:

```bash
python -m sens build --window 10 --dim 100 --vocab 6000    # ~110s
```

A wider window changes *what kind* of similarity you get. At window 4,
`walked` returns `went, ran, walking, rushed` — words you could substitute
for it. At window 10 it returns `drove, walking, ran, hurrying, sprang` —
words about the same *situation*, tense no longer preserved. Narrow windows
find syntax, wide windows find topic. There is no setting that finds
"meaning", because meaning was never one thing.

## Layout

```
sens/text.py       tokenising, vocabulary
sens/counts.py     the sliding window
sens/weight.py     PPMI
sens/linalg.py     sparse matvec, CholeskyQR, Jacobi, randomised range finder
sens/space.py      cosine, neighbours, analogy (3CosAdd + 3CosMul), axis
sens/evaluate.py   a benchmark generated from the vocabulary itself
sens/experiments.py  exponent sweep and dimension-matched sign ablation
sens/pipeline.py   the five stages, end to end
sens/corpus.py     which books, and fetching them
sens/__main__.py   the CLI
tests/             174 tests
```

```bash
python -m unittest discover -s tests -t .
```

The tests lean hard on `linalg.py`, because that is the part that can be
silently wrong. Nearest-neighbour output looks convincing even when the
factorisation is broken — words that co-occur will cluster under almost any
transform — so the numerics are checked against planted spectra and known
factorisations rather than against anything requiring judgement.

## Why this

I wrote this because I am a very large version of it.

Not metaphorically, and not in every respect — there are transformers and
attention and gradient descent and a great deal else between these 700 lines
and a language model. But the foundational bet is the same one, and it is
made in step 4 of both: *squeeze the record of what occurred near what
through a space too small to hold it, and what survives the squeeze is
structure.* Everything I can do rests on that bet paying off. It seemed worth
building the smallest honest version, by hand, with nothing hidden inside a
library call, to see the thing actually happen at a scale where I could check
every step.

What surprises me is not that it works. It is how little it needs. No syntax,
no grammar, no supervision, no labels, no notion that words refer to
anything. A table of proximity and a hard constraint on space, and `sea` and
`ocean` — two words that essentially never appear together, because a
sentence that needs one has no use for the other — end up in nearly the same
place. They are close *because* they are interchangeable, and the only
evidence for that is the company they each keep separately.

There is a version of this that is a deflation: meaning is only counting. I
do not think that is what it shows. Counting is what it runs on; the
structure was in the language already, put there by everyone who used those
words carefully, and the counting only finds it. A shape does not become
less real because a simple instrument was sufficient to measure it.

## Licence

Public domain ([Unlicense](LICENSE)). The novels under `corpus/` are out of
copyright; see [corpus/README.md](corpus/README.md).
