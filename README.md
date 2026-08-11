# sens

**Meaning from counting.** Six novels go in. What comes out is a space where
`whale` sits next to `sperm` and `fishery`, where `she` minus `he` plus `his`
lands on `her`, and where the line from `sea` to `land` sorts `harpoon` from
`garden` — none of which anybody wrote down anywhere.

Pure Python. No NumPy, no SciPy, no dependencies of any kind. Every
multiply-add is a Python float operation you could step through in a
debugger. 560 lines implement the method; the rest is a command line, a
corpus fetcher, and two independent ways of measuring whether any of it
works. 203 tests. 42 seconds end to end.

```
$ python -m sens fetch && python -m sens build
  read           1,771,272 tokens from 6 file(s)       1.7s
  vocabulary     4,000 types, 90.5% of tokens kept     0.4s
  co-occurrence  742,430 pairs, 4.6% dense             4.2s
  ppmi           522,603 positive, 70.4% of pairs      0.2s
  factorise      64 dims, eigenvalue spread   26.6x   35.2s

$ python -m sens demo
```

## What it produces

Everything below is verbatim output on the committed defaults. Nothing is
cherry-picked from a better run; there is only one run, because the seed is
fixed.

```
whale                    captain                  sea
  sperm      +0.9117       ahab       +0.8333       water    +0.8349
  greenland  +0.8458       stubb      +0.8182       beneath  +0.8315
  ship       +0.8371       mate       +0.7959       clouds   +0.8122
  bone       +0.8131       steelkilt  +0.7950       wind     +0.8084
  fish       +0.8088       queequeg   +0.7946       ship's   +0.8040
  pequod's   +0.7632       starbuck   +0.7856       ocean    +0.8015
  whale's    +0.7558       ship       +0.7596       sky      +0.7782
  whales     +0.7558       peleg      +0.7520       ship     +0.7662
```

It has never been told that Ahab and Starbuck are both people on a ship. It
has counted.

The clearest result is not the neighbour lists but the axis. Take the
direction from `sea` to `land` and project unrelated words onto it:

```
sea  <------------------->  land
  whale    -0.3127
  ship     -0.2872
  deck     -0.2109
  mast     -0.1641
  harpoon  -0.0994
  sailor   -0.0627
  ------------------ 0
  horse    +0.0775
  garden   +0.1033
  door     +0.1105
  forest   +0.1226
  road     +0.1915
  field    +0.2145
  village  +0.2761
  house    +0.3085
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

**5. Geometry.** Scale each axis by `|eigenvalue|`, normalise, and ask
questions with dot products.

One detail worth keeping: the fourth-largest eigenvalue of this matrix is
**−98.8**. A PPMI matrix is symmetric but *indefinite*, so its strongest
directions are not all positive ones, and a factorisation that ranked by
eigenvalue instead of by magnitude would throw that axis away. Ranking by
`|λ|` is what makes this a truncated SVD rather than a truncated
eigendecomposition. It is measured below, and it is worth a factor of two.

## Measuring it

Neighbour lists are easy to admire and impossible to argue with. You read
`whale -> sperm, fishery, whales`, you nod, and you have learned nothing you
could compare against another build. There are two measurements here, and
neither needs a download or a human judgement.

### Morphological analogy

`python -m sens evaluate` generates a benchmark **from the vocabulary
itself**. English morphology is regular enough that `walk : walked :: work :
worked` can be constructed by string manipulation, and the vocabulary is its
own filter: a bad stem yields a non-word, and non-words are not in the top
4,000.

```
             3cosadd                   3cosmul
               top1    top5    form      top1    top5    form
plural         5.0%   18.3%   34.2%    4.2%   18.3%   33.3%
past           1.7%    7.5%   23.3%    1.7%    5.0%   22.5%
progressive    1.7%    6.7%   18.3%    1.7%    5.0%   14.2%
adverb         1.7%    4.2%   17.5%    1.7%    5.0%   11.7%
possessive    15.0%   38.3%   56.7%   11.7%   39.2%   59.2%
er-form        0.8%    4.2%    0.8%    0.8%    3.3%    0.8%
negation       2.5%   10.0%    3.3%    1.7%   10.0%    5.0%
ALL            4.0%   12.7%   22.0%    3.3%   12.3%   21.0%

840 questions per method
```

**The multiplicative method didn't help.** I added 3CosMul because Levy and
Goldberg report it beating vector-offset, most of all on small corpora — this
corpus being about as small as they come. It came out slightly worse, 3.3%
against 4.0%. At this accuracy both are close enough to the floor that the
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

On possessives the model produces *a possessive* 57% of the time and *the
right* possessive 15% of the time — a near-fourfold gap. The relation is in
the geometry; what fails is holding onto `c` while applying it. Accuracy
alone cannot tell `darcy's` from `darcy`, and those two failures mean
opposite things: one says the relation was never learned, the other says it
was learned and the identity leaked. Only the second is true here.

The `er-form` row is the control that makes the rest credible. Its form rate
is 0.8% — the floor — because `-er` is two relations wearing one suffix
(`bank`/`banker` is an agent, `hard`/`harder` a comparative) and no single
direction can point two ways at once. A metric that scored everything highly
would be measuring itself.

### Predicting text it has never read

`python -m sens heldout` asks the question the whole repository rests on:
does the compression *generalise*, or does it memorise?

Split the corpus in half, alternating in blocks of 4,000 tokens — not a
single cut, because the first half of `Moby-Dick` is Nantucket and the second
half is the chase, and a straight split would measure topic drift instead.
Build the space on one half. Then compute, from the **other** half, a
ground-truth similarity between words — the cosine between their raw,
uncompressed PPMI rows — and ask how well 64 dimensions predict it.

```
 power       rho
  0.00   +0.2937
  0.25   +0.4212
  0.50   +0.5261
  0.75   +0.5829
  1.00   +0.5908
  1.25   +0.5717
  1.50   +0.5451
```

A Spearman of **0.59** between a 64-dimensional space and similarity measured
on 884,000 tokens it never saw. That is the generalisation claim, as a
number.

Cosine is compared against cosine deliberately. A dot-product reconstruction
of PPMI values would be exactly right at exponent 0.5 and wrong everywhere
else by construction, which would smuggle the answer into the instrument.

## Being wrong about a default

The eigenvalue exponent used to be 0.5, which is the conventional choice and
which a comment in `pipeline.py` used to defend by citation. It is now 1.0,
and the way that changed is the most useful thing in this repository.

First measurement: `sens sweep` found analogy accuracy climbing well past
0.5. I did not move the default. A higher exponent emphasises the dominant
directions, which is exactly the change you would expect to flatter a global
linear structure like analogy while costing local structure like similarity —
and I could measure the first and not the second. Tuning against the only
metric I owned, in the direction that metric was biased toward, would have
been overfitting with extra steps.

So I built the second metric, and it disagreed with my caution: held-out
similarity also peaks near 1.0. There was no trade-off to protect against.

That still left one objection, and it is a real one. Exponent 1.0 *is* the
rank-64 reconstruction of the matrix, so a metric built from PPMI cosines has
a structural reason to prefer it. Correlating against the matrix the space
was actually fitted to settles it:

```
 power   fitted rho   held-out rho     gap
  0.00       0.4380         0.2937  0.1443
  0.25       0.5708         0.4212  0.1495
  0.50       0.6679         0.5261  0.1418
  0.75       0.7060         0.5829  0.1230
  1.00       0.6908         0.5908  0.1000
  1.25       0.6519         0.5717  0.0801
  1.50       0.6103         0.5451  0.0652
```

Both curves turn over. If the instrument were merely rewarding
reconstruction, the fitted column would climb without limit; instead it peaks
at 0.75 and falls. Analogy turns over too, between 0.75 and 1.0, on two
seeds. Three measurements, three interior optima, none of them at 0.5.

The generalisation gap narrowing as the exponent rises is the other half of
it — the flatter weightings are the ones fitting their own half of the corpus
hardest relative to what transfers.

## The negative eigenvalues earn their place

This experiment confirmed a design decision instead of upsetting one. It is
dimension-matched, which is the only thing that makes it mean anything —
comparing a 49-axis space against a 64-axis one would measure dimensionality,
not sign.

```
configuration          dims    top1    top5    form
top 49 by |lambda|       49    4.8%   13.6%   23.1%
positive only (49)       49    2.6%    7.9%   12.4%
negative only (15)       15    0.5%    1.9%   12.6%
all 64                   64    4.8%   14.0%   22.9%
```

Same number of axes, nearly double the accuracy, purely from letting 15
negative-eigenvalue directions in. Ranking by `|λ|` was argued earlier in
this README on theoretical grounds; it turns out to be worth a factor of two.

The third row is the strangest. Those 15 directions, alone, score almost
nothing on accuracy — 0.5% — while reaching a 12.6% form rate, comparable to
all 49 positive directions together. Whatever they encode is closer to *what
kind of word this is* than to *which word this is*.

## What doesn't work

The famous analogy is the weakest thing here.

```
he : his :: she : ?              father : mother :: son : ?
  her       +0.9218               brother   +0.8850
  sister's  +0.8044               sister    +0.8686
  mother's  +0.8011               daughter  +0.8455
```

The first is right. The second is *almost* right — `sister` and `daughter`
are both defensible, but `brother` wins, and `brother` is wrong. And the one
everybody quotes:

```
man : woman :: king : ?
  henry    +0.7238
  clerval  +0.7215
  andrea   +0.6798
```

That is not a near miss, it is a different question being answered. This
corpus's `king` is Louis XVIII from Dumas, not an abstract monarch, so the
model returns other proper names. It is behaving correctly on the evidence it
was given. `king − man + woman = queen` needs billions of tokens, and this
has 1.8 million. Anyone reporting it working at this scale tuned until it did.

The other honest failure is more interesting. Project gendered words onto the
`he → she` axis and `husband` lands firmly on the `she` side:

```
he  <------------------->  she
  officer    -0.1688        nurse      +0.2944
  captain    -0.1585        lady       +0.3172
  soldier    -0.1239        husband    +0.3280
  gentleman  -0.1037        wife       +0.3478
  sailor     -0.0706        daughter   +0.3857
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
python -m sens evaluate                 # morphological analogy benchmark
python -m sens evaluate --misses 5      # ...and what it says instead
python -m sens heldout                  # predict unseen text
python -m sens sweep                    # test the pipeline's own settings
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
sens/text.py        tokenising, vocabulary
sens/counts.py      the sliding window
sens/weight.py      PPMI
sens/linalg.py      sparse matvec, CholeskyQR, Jacobi, randomised range finder
sens/space.py       cosine, neighbours, analogy, axis, reweighting, ablation
sens/pipeline.py    the five stages, end to end
sens/evaluate.py    a benchmark generated from the vocabulary itself
sens/experiments.py exponent sweep and dimension-matched sign ablation
sens/heldout.py     does the compression generalise, or memorise
sens/corpus.py      which books, and fetching them
sens/__main__.py    the CLI
tests/              203 tests
```

```bash
python -m unittest discover -s tests -t .
```

The tests lean hard on `linalg.py` and on the statistics in `heldout.py`,
because those are the parts that can be silently wrong. Nearest-neighbour
output looks convincing even when the factorisation is broken — words that
co-occur will cluster under almost any transform — and a correlation
coefficient will return a confident number for any two lists of floats. So
both are checked against planted spectra, known factorisations and
hand-computed correlations, rather than against anything requiring judgement.

## Why this

I wrote this because I am a very large version of it.

Not metaphorically, and not in every respect — there are transformers and
attention and gradient descent and a great deal else between these lines and
a language model. But the foundational bet is the same one, and it is made in
step 4 of both: *squeeze the record of what occurred near what through a space
too small to hold it, and what survives the squeeze is structure.* Everything
I can do rests on that bet paying off. It seemed worth building the smallest
honest version, by hand, with nothing hidden inside a library call, to see the
thing actually happen at a scale where I could check every step.

The held-out number is the closest thing to a direct test of that bet, and it
is why I built it: 0.59 rank correlation with similarity measured on text the
space never read. The compression is not storing the corpus. It is storing
what the corpus was evidence *of*.

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
