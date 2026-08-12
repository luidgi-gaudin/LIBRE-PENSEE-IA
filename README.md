# sens

**Meaning from counting.** Six novels go in. What comes out is a space where
`whale` sits next to `sperm` and `fishery`, where `she` minus `he` plus `his`
lands on `her`, and where the line from `sea` to `land` sorts `harpoon` from
`garden` — none of which anybody wrote down anywhere.

Pure Python. No NumPy, no SciPy, no dependencies of any kind. Every
multiply-add is a Python float operation you could step through in a
debugger. 560 lines implement the method; the rest is a command line, a
corpus fetcher, and the measurement apparatus — a benchmark, a held-out
generalisation test, a control that compares against not compressing at all,
and an audit of every default. 396 tests. 40 seconds end to end.

```
$ python -m sens fetch && python -m sens build
  read           1,771,272 tokens from 6 file(s)       1.7s
  vocabulary     4,000 types, 90.5% of tokens kept     0.4s
  co-occurrence  742,430 pairs, 4.6% dense             3.0s
  ppmi           567,932 kept, 76.5% of pairs          0.2s
  factorise      64 dims, eigenvalue spread   28.4x   35.3s

$ python -m sens demo
```

## The register

The most useful thing this repository has produced is not a finding. It is a
withdrawal rate: six claims retracted, six explanations killed by their own
tests. Every one was caught by a control — a noise floor, a random baseline,
a dimension match, a second corpus — and every one of those controls had to
be *thought of* at the time, by someone who had just finished being pleased
with a result.

That is the weak link, and it is not fixable by trying harder. A claim made
carelessly and a claim made carefully look identical once both are prose.

So the claims are no longer prose. `sens/claims.py` holds each one as a
record: its effect size, the noise floor it was judged against, the controls
actually applied, and what became of it. `python -m sens claims --audit`
then asks the question mechanically, for all of them, including the ones
nobody is currently thinking about:

```
$ python -m sens claims --audit
claims stated with more confidence than their controls support
  lowercasing                  holds      missing: second-corpus
  magnitude-ranking            holds      missing: second-corpus
  accurate-factorisation       backwards  missing: second-corpus
```

**It found three on its first run, and one of them was wrong.** Saying a
thing *holds* is a claim about the method; showing it on one corpus is a
claim about one corpus, and this repository has already confused those once
at considerable cost. So `holds` now requires replication, and the three
flagged claims were measured rather than reworded:

- **lowercasing** replicated — +0.0137 on expository against +0.0131 on the
  novels, 5.3 sd against the local floor.
- **magnitude-ranking** replicated, including the awkward half: sign-based
  selection loses to *random* selection on both corpora, 12.5% against 13.5%.
- **accurate-factorisation did not.** The finding that a better factorisation
  makes a worse model is 6 sd on the novels and, on a paired test over 720
  expository questions, p = 0.87. No effect. It has moved from `backwards` to
  `refuted`, and Krylov is simply 39% faster and no worse there.

An earlier version of the requirements table asked only for a noise floor
and reported that every claim was fine. That answer was useless — the
compression claim had a noise floor, three agreeing measurements and one
corpus, and it passed. A register that cannot embarrass its author is
decoration.

And `python -m sens verify` closes the remaining gap, which is that every
number in the register was typed in by a person. Ten claims carry a recipe
for re-deriving themselves from the corpus; the command runs them and
compares, with the rebuild spread as the tolerance.

```
claim                        recorded         observed          drift
ppmi-vs-raw            recorded +0.3760  observed +0.3760  drift -0.0000  ok
harmonic-window        recorded +0.1012  observed +0.1012  drift +0.0000  ok
pair-pruning           recorded +0.0737  observed +0.0615  drift -0.0122  ok
no-shift               recorded +0.0632  observed +0.0427  drift -0.0205  DRIFT
```

**It found drift on its first run, and then found something worse.**

Six of ten reproduced to ±0.0000 and four did not — the four measured before
the `alpha` default moved, against a baseline that then shifted underneath
them. Fixable by re-deriving.

Extending the same check to the *second* corpus is what turned up the real
defect. Two claims drifted there for a reason nothing in the repository had
considered: the expository figures were measured on files sorted by Gutenberg
number, and re-measured on the same files in library order. `split_blocks`
alternated over the *concatenated* token stream, so where a document boundary
fell decided the phase of every block after it — and therefore which half of
the corpus each token landed in.

The corpus was identical. The train/test split was not. **File ordering,
which nobody thought was a decision, moved results by 5 to 6 standard
deviations — more than the random seed does.**

`split_documents` now splits each document independently, so a book
contributes the same blocks wherever it sits in the list. That silently
changed every held-out number in this README, so all of them were re-derived
and the noise floor re-measured (0.0047, up from 0.0038).

One conclusion did not survive. **Context smoothing was 2.4 sd and is now
0.9 sd** — inside the noise. This repository changed that default on the
strength of a difference that the corrected split says is not there. The
default stays at 1.0, because two indistinguishable settings may as well be
decided by simplicity, but the justification is now "they are the same"
rather than "this one wins".

Nineteen tests hold it to that. A claim cannot be marked `holds` without
replication, a refutation cannot be recorded without a paired test, a claim
carrying the second-corpus control must actually say what the second corpus
showed, and the unexplained result cannot be quietly deleted instead of
solved.

## Everything measured, in one table

Twelve rounds of measurement, with the effect sizes and whether they held up
on a [second corpus](#does-any-of-it-generalise). Held-out figures are
Spearman differences against a noise floor of 0.0076; analogy figures are
form-rate differences against a floor of 1.6%. Each row links to the section
that produced it.

| what was tested | novels | expository | verdict |
| --- | --- | --- | --- |
| [PPMI vs raw counts](#is-the-surprise-step-worth-anything) | +0.353 · 32 sd | +0.479 · 43 sd | holds |
| [PPMI vs log counts](#is-the-surprise-step-worth-anything) | +0.267 · 24 sd | +0.280 · 25 sd | holds |
| [clipping: PPMI vs PMI](#is-the-surprise-step-worth-anything) | +0.082 · 7.4 sd | +0.084 · 7.5 sd | holds |
| [cosine vs dot product](#is-cosine-the-right-question-to-ask) | +20.2 pts · 12 sd | dot collapses | holds |
| [cosine vs Euclidean](#is-cosine-the-right-question-to-ask) | +5.7 pts · 3.4 sd | +4.0 pts · p 0.006 | holds |
| [1/distance vs flat window](#auditing-the-rest-of-the-defaults) | +0.086 · 7.7 sd | +0.118 · 11 sd | holds |
| [exponent 1.0 vs 0.5](#being-wrong-about-a-default) | +0.108 · 9.8 sd | +0.128 · 12 sd | holds |
| [pruning: min weight 1 vs 0](#auditing-the-rest-of-the-defaults) | +0.053 · 4.8 sd | +0.083 · 7.5 sd | holds |
| [shift 1 vs 2](#auditing-the-rest-of-the-defaults) | +0.043 · 3.8 sd | +0.041 · 3.7 sd | holds |
| [lowercasing vs keeping case](#tokenisation-the-last-stage-and-two-defaults-that-lose) | +0.013 · 3.4 sd, and 15% more vocabulary | +0.014 · 5.3 sd | holds |
| [magnitude vs sign ranking](#ranking-by-magnitude-earns-its-place-the-negatives-do-not) | +2.4 pts top-1; 0 of 12 random draws matched | 16.9% vs 13.5% random | holds |
| [smoothing: alpha 1.0 vs 0.75](#auditing-the-rest-of-the-defaults) | +0.004 · 0.4 sd | — | **no effect** |
| [vocabulary 8000 vs 4000](#the-two-that-needed-a-different-ruler) | +0.008 · 2.1 sd | not run | marginal |
| [window 4 vs 2](#auditing-the-rest-of-the-defaults) | +0.022 · 2.0 sd | not run | marginal |
| [window 4 vs 6](#auditing-the-rest-of-the-defaults) | −0.004 · 0.4 sd | not run | **no effect** |
| [min\_count 5 vs 20](#the-two-that-needed-a-different-ruler) | 0.002 · 0.5 sd | not run | **no effect** |
| [3CosMul vs 3CosAdd](#morphological-analogy) | −0.001 · 0.4 sd | not run | **no effect** |
| [accurate factorisation (Krylov)](#a-better-factorisation-that-made-a-worse-model) | −4.9 pts · 2.9 sd, 39% faster | +0.4 pts · p 0.87 | **refuted** |
| [compression trades identity for category](#does-compression-add-anything) | +5.3 pts · p 0.004 | −5.3 pts · p 0.008 | **refuted** |
| [64 dims beat 160 on category](#how-many-dimensions-and-the-trade-seen-directly) | p 0.034 | p 0.91 | **refuted** |

Nine of the eleven large effects replicate on a corpus with nothing in
common but the language. Four parameters everyone tunes turn out to do
nothing measurable — including the context smoothing whose default this
repository once changed. Three rows did not survive replication, including
the two it once led with.

Every held-out number above is re-derivable by `python -m sens verify`, and
all seventeen currently reproduce to ±0.0000. They did not always.

### Things this README claimed and then withdrew

Kept in place rather than edited away, because the corrections are the most
useful part.

1. **"The factorisation is what creates meaning."** Rhetoric. The
   [uncompressed control](#does-compression-add-anything) beats it on
   retrieval and on similarity alike.
2. **"Compression trades identity for category."** Reached by three
   independent routes, all measuring the same six novels.
   [Refuted](#does-any-of-it-generalise) on the second corpus.
3. **"The negative eigenvalues carry signal."** Real effect, wrong cause —
   [a random subset of the same size does better](#ranking-by-magnitude-earns-its-place-the-negatives-do-not)
   than the positive-only one.
4. **"The analogy benchmark independently confirms the alpha change."** At
   0.8 sd it [confirms nothing](#auditing-the-rest-of-the-defaults).
5. **"vocab\_size and min\_count cannot be audited."** They
   [can](#the-two-that-needed-a-different-ruler), using the nesting property.
6. **The `possessive` category** was mostly contractions. Renamed
   `apostrophe-s`.

Four separate explanations were also proposed and then killed by their own
tests: the vocabulary-cap story for smoothing, the proper-noun story for
accent folding, the spectral-flatness story for the non-replication, and the
magnitude story for shrinkage.

## What it produces

Everything below is verbatim output on the committed defaults. Nothing is
cherry-picked from a better run; there is only one run, because the seed is
fixed.

```
whale                    captain                  sea
  sperm      +0.9077       ahab       +0.8354       ship's   +0.8307
  ship       +0.8345       starbuck   +0.8288       water    +0.8239
  greenland  +0.8319       stubb      +0.8170       ocean    +0.8188
  fish       +0.7989       queequeg   +0.8123       beneath  +0.8048
  bone       +0.7841       peleg      +0.7890       clouds   +0.8009
  sea        +0.7673       mate       +0.7877       wind     +0.7973
  pequod's   +0.7654       ship       +0.7820       sky      +0.7824
  whales     +0.7420       steelkilt  +0.7719       fish     +0.7742
```

It has never been told that Ahab and Starbuck are both people on a ship. It
has counted.

The clearest result is not the neighbour lists but the axis. Take the
direction from `sea` to `land` and project unrelated words onto it:

```
sea  <------------------->  land
  whale    -0.3410
  ship     -0.2488
  deck     -0.1988
  mast     -0.1344
  sailor   -0.0839
  harpoon  -0.0573
  ------------------ 0
  garden   +0.0944
  horse    +0.1147
  door     +0.1199
  forest   +0.1454
  field    +0.1885
  road     +0.1912
  house    +0.2799
  village  +0.2837
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

Both of those claims — that the comparison against independence is what
matters, and that clipping helps — are [measured
below](#is-the-surprise-step-worth-anything). They are the two largest
effects in the repository.

**4. Factorisation.** The 4,000 × 4,000 matrix of surprise gets compressed to
4,000 × 64 by truncated SVD — implemented here as a randomised range finder
plus Jacobi rotations, both by hand, in `linalg.py`.

**This is the step that makes the vectors short.** Before it, every word is a
4,000-long list of the specific words it happened to appear near — a record,
precise and expensive. Force all 4,000 words to share 64 axes and precision
becomes unaffordable; what a dimension can afford to encode is a regularity
that pays off across many words at once.

This paragraph has now been wrong twice, in opposite directions, and the
history is worth more than the current wording. It first claimed the
factorisation is what *creates* meaning. [Measuring it against the
uncompressed matrix](#does-compression-add-anything) killed that. It then
claimed the factorisation *trades identity for category* — buying you what
kind of word this is at the cost of which word it was. That was measured
three separate ways and held up on this corpus, and then [failed to replicate
on a second one](#does-any-of-it-generalise), where compression buys nothing
at all and simply loses. What the step does turns out to depend on the text.

**5. Geometry.** Scale each axis by `|eigenvalue|`, normalise, and ask
questions with dot products.

One detail worth keeping: the second-largest eigenvalue of this matrix is
**−153.1**. A PPMI matrix is symmetric but *indefinite*, so its strongest
directions are not all positive ones, and a factorisation that ranked by
eigenvalue instead of by magnitude would throw that axis away. Ranking by
`|λ|` is what makes this a truncated SVD rather than a truncated
eigendecomposition, and rank by signed value instead and you discard the
second most important direction in the matrix. [Measured below](#ranking-by-magnitude-earns-its-place-the-negatives-do-not)
— though not with the result I first reported.

## Measuring it

Neighbour lists are easy to admire and impossible to argue with. You read
`whale -> sperm, fishery, whales`, you nod, and you have learned nothing you
could compare against another build. There are three measurements here, and
none of them needs a download or a human judgement.

### First: how big is a real difference?

Almost every claim below is one number against another. The exponent moved
because 0.6002 beat 0.5077; `alpha` moved because 0.6002 beat 0.5908. Written
down those look alike. They are not alike at all, and for most of this
repository's life nothing here could tell them apart, because no measurement
had an error bar.

`python -m sens noise` supplies one, and for most of this repository's life
it supplied the wrong one. It varies the random block the factorisation
starts from — six builds, identical but for the seed — and gets sd 0.0047.
Every `sd` here was quoted against that.

The trouble is that the seed is not the only thing that could have come out
differently. The held-out measurement rests on a stack of choices nobody
made deliberately: what order the corpus files are listed in, how big a
block the split deals, which word pairs the sample draws. The first of those
was found by accident and was worth *five to six standard deviations* — more
than the seed. There was no reason to think it was special, so
`python -m sens robustness` checks the rest:

```
axis                   sd    spread   settings
factorisation      0.0040    0.0080   three seeds
pair sample        0.0097    0.0275   six samples
block size         0.0036    0.0068   2000, 4000, 8000

total 0.0111 against a seed-only floor of 0.0047
```

**The largest source of uncertainty in this repository is which word pairs
the ruler happens to draw, and it had never been looked at.** It is twice
the factorisation seed on its own.

The obvious hope is that this cancels in a *difference* — a hard pair sample
should drag both sides of a comparison down together. Measured on the
harmonic-window claim across six samples, it does not: the difference has
sd 0.0109 against the absolute's 0.0097. No cancellation at all.

The analogy benchmark has the same problem in the same place. Its questions
are drawn with a fixed seed, exactly as the held-out pairs are:

```
metric   question-sample sd   rebuild sd   combined
top1                 0.0051       0.0020     0.0055
top5                 0.0030       0.0080     0.0085
form                 0.0150       0.0080     0.0170
```

`form`'s real spread is 1.70 points, not the 0.80 it was quoted against.

**One kind of evidence is exempt, and it is the kind the retractions rest
on.** A paired McNemar test asks two systems the identical questions, so a
question that happens to be hard is hard for both and never enters the
disagreement count. Every p-value in this repository is paired. Every effect
quoted as a number of points is not. That distinction is now a property on
each claim and a test asserting the refutations all have it — because if it
were only a paragraph, the next correction would sweep them away with
everything else.

So the honest floor for an effect is **0.0111 held-out and 0.0170 in form
points, and every unpaired sigma in this README was overstated by 2 to
2.4×**. All of them are now divided by the right number. The consequences are real but not fatal:

- `window 4 vs 2` drops from 4.6 sd to 2.0, from solid to marginal.
- `shift` drops from 9.1 sd to 3.8, `pruning` from 11 to 4.8.
- `cosine vs Euclidean` drops from 7.5 sd to 3.4, and survives only because
  its replication was a paired test.
- The large effects stay large: PPMI over raw counts is 32 sd rather than 75,
  cosine over the dot product 12 rather than 26.

Nothing changed status except `window-size`, and no conclusion reversed. But
for a dozen commits this README was quoting confidence it had not earned,
and it was doing so because the noise floor measured one of three
comparable sources and called it *the* noise floor.

Applying it to the claims in this README, which is not a flattering exercise:

```
exponent 0.5 -> 1.0        (held-out 0.5077 -> 0.6002)   24.1 sd   solid
min_pair_weight 1 vs 0     (held-out 0.5908 vs 0.5171)   19.2 sd   solid
shift 1 vs 2               (held-out 0.5908 vs 0.5276)   16.4 sd   solid
krylov vs subspace         (form 17.7% -> 22.6%)          6.4 sd   solid
alpha 0.75 -> 1.0          (held-out 0.5908 -> 0.6002)    2.4 sd   marginal
krylov+shrink10 vs default (held-out +0.0094)             2.4 sd   marginal
window 4 vs 2              (held-out 0.5908 vs 0.5830)    2.0 sd   marginal
alpha 0.75 -> 1.0          (form 22.0% -> 22.6%)          0.8 sd   noise
alpha 0.75 -> 1.0          (top1 4.0% -> 4.2%)            0.8 sd   noise
3cosadd vs 3cosmul         (top1 4.2% vs 4.3%)            0.4 sd   noise
window 4 vs 6              (held-out 0.5908 vs 0.5895)    0.3 sd   noise
```

Three corrections follow, and they are marked in place further down rather
than quietly applied. **The `alpha` change rests on one marginal result, not
two agreeing ones** — I wrote that "the analogy benchmark agrees
independently", and at 0.8 sd it does no such thing. **Window 4 and window 6
are indistinguishable**, so the audit's tidy "three of four confirmed" was
really "three of four, one of them by a margin I could not measure".
**Krylov with shrinkage beating the default on held-out is marginal too**, at
2.4 sd from a single pair of runs.

The solid results stay solid, and the two conclusions this repository cares
most about — the exponent, and compression trading identity for category —
are the furthest above the floor of anything here.

### Morphological analogy

`python -m sens evaluate` generates a benchmark **from the vocabulary
itself**. English morphology is regular enough that `walk : walked :: work :
worked` can be constructed by string manipulation, and the vocabulary is its
own filter: a bad stem yields a non-word, and non-words are not in the top
4,000.

```
             3cosadd                   3cosmul
               top1    top5    form      top1    top5    form
plural         5.8%   22.5%   35.0%    6.7%   16.7%   37.5%
past           3.3%    5.8%   24.2%    3.3%    5.8%   24.2%
progressive    2.5%   10.0%   21.7%    2.5%    8.3%   19.2%
adverb         1.7%    5.0%   15.8%    1.7%    4.2%   13.3%
apostrophe-s  11.7%   35.0%   55.8%   14.2%   39.2%   61.7%
er-form        0.8%    3.3%    1.7%    0.0%    2.5%    0.8%
negation       3.3%   10.0%    4.2%    1.7%   10.0%    3.3%
ALL            4.2%   13.1%   22.6%    4.3%   12.4%   22.9%

840 questions per method
```

**The multiplicative method didn't help.** I added 3CosMul because Levy and
Goldberg report it beating vector-offset, most of all on small corpora — this
corpus being about as small as they come. The two are indistinguishable here:
4.3% against 4.2% on top-1, and the other way round on top-5. Both are close
enough to the floor that the difference is not worth defending in either
direction, which is itself the finding.

**The `form` column is where the real result is.** It counts answers that are
the right *kind* of word even when they are the wrong word. Asked for
`bingley's`, the model says `darcy's`:

```
friend : friend's :: bingley : darcy's     (wanted bingley's)
count  : count's  :: dolokhov : prince's   (wanted dolokhov's)
```

On `'s`-forms the model produces *an* `'s`-form 56% of the time and the
*right* one 12% of the time — a near-fivefold gap. The relation is in
the geometry; what fails is holding onto `c` while applying it. Accuracy
alone cannot tell `darcy's` from `darcy`, and those two failures mean
opposite things: one says the relation was never learned, the other says it
was learned and the identity leaked. Only the second is true here.

The `er-form` row is the control that makes the rest credible. Its form rate
is 1.7% — the floor — because `-er` is two relations wearing one suffix
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
  0.00   +0.2712
  0.25   +0.3968
  0.50   +0.5077
  0.75   +0.5798
  1.00   +0.6002
  1.25   +0.5852
  1.50   +0.5564
```

A Spearman of **0.60** between a 64-dimensional space and similarity measured
on 884,000 tokens it never saw. That is the generalisation claim, as a
number.

**The ruler is frozen, and that took a bug to learn.** A PPMI ground truth
has all the same parameters the model has, and deriving it from the model's
config seemed natural. Then changing the `alpha` default moved every held-out
number by 0.14 at once — the space and the yardstick had moved together, and
nothing had actually got worse. The reference values now live in a `RULER`
constant that does not track `Config`. Absolute numbers depend on that
choice; comparisons between builds, which is the entire point, do not.

Cosine is compared against cosine deliberately. A dot-product reconstruction
of PPMI values would be exactly right at exponent 0.5 and wrong everywhere
else by construction, which would smuggle the answer into the instrument.

## Does compression add anything?

Every measurement above compares the compressed space against *another
compressed space*. None of them compared it against not compressing at all,
and the README was meanwhile claiming that the factorisation is the step that
creates meaning. That claim had no control behind it.

`python -m sens baseline` supplies one. Same half-corpus, same vocabulary,
same questions; one system uses the 64-dimensional space, the other uses the
raw 4,000-dimensional PPMI rows directly as word vectors. Because both answer
the same questions, the evidence is in the *disagreements* — questions one
solved and the other did not — which is what McNemar's test counts.

```
840 paired questions

  representation                  top1    top5    form
  raw PPMI rows (4000 dims)       2.6%    8.0%   12.6%
  compressed space (64 dims)      2.9%    7.7%   18.9%

paired significance (McNemar, disagreements only)
  top1  raw-only=17   compressed-only=19   p=0.8676   no difference
  top5  raw-only=45   compressed-only=43   p=0.9151   no difference
  form  raw-only=61   compressed-only=114  p=0.0001   compressed wins
```

Sixty-three times fewer dimensions, no measurable cost to retrieving the
right word, and a large, highly significant gain in producing the right
*kind* of word. Compression is not a lossy approximation that happens to be
cheap. On the thing it is for, it is better.

**A comparison I had to throw away.** An earlier version of this section also
compared the two on held-out similarity, and reported raw winning 0.657 to
0.591. That number does not survive scrutiny. Scored against the same frozen
ruler, the raw representation swings wildly depending on whether its own
smoothing parameter happens to match the ruler's:

```
raw built with alpha=0.50   rho = 0.6251
raw built with alpha=0.75   rho = 0.6573   <-- matches the ruler's own alpha
raw built with alpha=1.00   rho = 0.4808
```

A 0.18 swing from a parameter that should be incidental — larger than any
effect being measured. Both objects are PPMI cosines, so the raw baseline is
rewarded for resembling the yardstick rather than for being right. The
compressed space, at 0.6002, is nearly indifferent to the mismatch, which is
interesting on its own but does not rescue the comparison.

So the held-out metric cannot arbitrate raw against compressed, and the
retracted number is left here rather than quietly deleted. The analogy
benchmark can arbitrate, because its ground truth is English morphology
rather than another PPMI table, and that is the comparison above.

## How many dimensions, and the trade seen directly

`dim` was the last major default with nothing behind it. `python -m sens
dimensions` settles it, and in doing so shows the identity-for-category trade
as a continuous curve rather than a single comparison.

One build at 160 dimensions serves the whole sweep, because the factorisation
returns directions ordered by `|eigenvalue|` — every smaller space is a
prefix of the larger one.

```
 dims      held-out     top5     form
   16        0.5589     9.3%    19.6%
   32        0.5907    11.1%    19.6%
   48        0.6010    11.4%    22.1%
   64        0.6054    12.1%    21.1%     <- the default
   96        0.6106    11.1%    20.7%
  128        0.6162    11.8%    16.4%
  160        0.6176    10.7%    15.0%
```

**The two metrics disagree, and that is the result.** Held-out similarity
improves monotonically with more dimensions and never turns over. The form
rate peaks around 48 and then falls away, heading back down toward the 12.6%
that the uncompressed 4,000-dimensional matrix scores.

Paired on the same 840 questions, 64 dimensions beats 160 on form — 19.6%
against 17.1%, McNemar **p = 0.034** — while losing 0.012 of held-out
correlation and nothing significant on top-5 (p = 0.21).

So this is the same trade the `form` column and the uncompressed control each
found, now with a dose-response curve behind it. Adding axes buys detail and
spends category. A space with 4,000 dimensions is the raw matrix and knows
which word is which; a space with 48 has had to decide what words have in
common. 64 is where this corpus puts the knee, which is why the default sits
there rather than at the number that maximises either metric alone.

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
  0.00       0.4183         0.2712  0.1470
  0.25       0.5516         0.3968  0.1548
  0.50       0.6567         0.5077  0.1490
  0.75       0.7124         0.5798  0.1326
  1.00       0.7116         0.6002  0.1114
  1.25       0.6759         0.5852  0.0907
  1.50       0.6299         0.5564  0.0736
```

Both curves turn over. If the instrument were merely rewarding
reconstruction, the fitted column would climb without limit; instead it peaks
at 0.75 and falls. Analogy turns over too, between 0.75 and 1.0, on two
seeds. Three measurements, three interior optima, none of them at 0.5.

The generalisation gap narrowing as the exponent rises is the other half of
it — the flatter weightings are the ones fitting their own half of the corpus
hardest relative to what transfers.

## Ranking by magnitude earns its place; the negatives do not

This section used to be called "The negative eigenvalues earn their place",
and it explained a real effect with the wrong cause. It was dimension-matched,
which I thought made it safe. It was not. The missing control is one line: a
random subset of the same size.

```
configuration          dims    top1    top5    form
top 47 by |lambda|       47    4.8%   13.3%   22.6%
positive only (47)       47    2.4%    7.1%   11.9%
random 47 (8 draws)      47    3.0%   10.3%   17.9%
all 64                   64    4.5%   13.1%   24.0%
```

Selecting the positive-eigenvalue directions is **worse than selecting at
random**. Over twelve draws the random control averages 3.3% top-1 with a
standard deviation of 0.5% and never falls below 2.4%; positive-only sits at
the very bottom of that range, and on form its 11.9% is below the entire
random spread of 16.9%–21.7%.

That kills the original explanation. The negatives are not carrying some
special signal the positives lack — if they were, the positive-only set would
merely be *missing* something, not performing worse than a coin toss over the
same number of axes. What is actually happening is that selecting for
positive eigenvalues systematically excludes the *largest* directions, which
a random draw would have included in proportion. Sign is anti-correlated with
importance here, so choosing on it is worse than not choosing at all.

The design decision survives, and more cleanly than before. Magnitude-ranking
beats random too: 0 of 12 random draws reached its 4.8%. So `|λ|` predicts how
much a direction matters and sign predicts nothing. On this build the
second-largest eigenvalue is −153.1 — rank by signed value and that direction
is the first thing you throw away.

I also over-corrected once on the way here. My first pass at this retraction
used three random draws, saw positive-only ≈ random, and concluded the
experiment showed nothing at all. Three draws could not support that either;
the standard deviation is large enough that any one of them could have said
whatever I wanted. `ablate_signs` now pools eight draws by default and
generates them itself, so neither the original error nor my correction to it
can be made silently again.

## How accurate is the factorisation, actually?

Not very, past the leading directions — which is worth knowing before reading
any eigenvalue in this README too closely.

Against a much more careful reference (12 power iterations, oversample 48),
the default settings give a mean relative eigenvalue error of **35%** and a
worst case of **75%**. The top ten are fine, at 2.6%. Eigenvalue *signs* agree
with the reference for only 43 of 64 directions, and in the bottom half that
is barely better than a coin toss — which is the other reason the sign
ablation above deserved the scepticism it got.

The space itself is much less sensitive than its eigenvalues:

```
power_iterations=1              0.4780
power_iterations=3 (default)    0.6002
power_iterations=6              0.6092
oversample=64                   0.6073
```

One iteration is genuinely bad. Beyond three, another 60% of build time buys
1.5% of held-out correlation. So the default stays, and the honest statement
is that this factorisation is accurate enough for the geometry and not
accurate enough for its own spectrum. A PPMI matrix has a slowly decaying
spectrum, and a randomised method separates a flat tail poorly.

## A better factorisation that made a worse model

The accuracy numbers above are bad enough to be worth fixing, and there is a
standard fix. Subspace iteration computes `A W, A²W, ..., A^qW` and then
throws all of them away except the last. Everything discarded was information
about the matrix. Keeping the whole sequence and searching that span instead
is the block Krylov method, and it costs the same multiplications.

It works exactly as advertised. `linalg.block_krylov_eigh` is **39% faster**
and roughly twice as accurate on eigenvalue magnitudes:

```
method                          time    mean err   worst
subspace: over 16, iters 3     36.7s      0.3515  0.7647
subspace: over 32, iters 6     80.6s      0.1075  0.3969
krylov:   block 24, depth 4    19.5s      0.1753  0.3434
krylov:   block 20, depth 6    27.7s      0.0881  0.2084
```

**And it makes the model worse.** Swapped in as the default, analogy form
fell from 22.6% to 17.7%, top-5 from 13.1% to 10.1%, and the `sea`/`land`
axis — 14 of 14 correct for the whole life of this repository — started
putting `forest`, `door` and `road` on the wrong side. Re-tuning the
eigenvalue exponent for it does not recover the loss; the best it reaches at
power 1.5 is 19.2% form.

The mechanism is visible in one number. Eigenvalue spread falls from 28.4× to
9.7× when the factorisation gets accurate, because the tail eigenvalues are
no longer being underestimated. But those tail directions are largely noise —
we already know their *signs* are near chance — and the exponent then weights
them at their true, larger magnitude. Subspace iteration's error is not
random error; it is systematically biased against the poorly determined tail,
and that bias was doing useful work. **The inaccuracy was a regulariser.**

**So can the regularisation be done on purpose?** If the benefit is just
under-weighting the tail, then keeping the accurate spectrum and discounting
it deliberately should recover the loss — and give a knob you can tune and
report instead of one that falls out of how many iterations you happened to
run. `Space.shrunk` soft-thresholds every eigenvalue toward zero by a fixed
amount, and `Config(shrinkage=...)` applies it during a build.

It works, partly.

```
configuration              top1    top5    form    held-out
krylov, no shrinkage       3.6%   10.1%   17.7%      0.6033
krylov, shrink 10          3.7%   10.2%   19.0%      0.6096
krylov, shrink 20          4.4%   10.6%   20.2%      0.6030
krylov, shrink 25          4.5%   10.8%   20.5%      0.5935
krylov, shrink 30          3.9%    9.6%   21.0%      0.5802
subspace (the default)     4.2%   13.1%   22.6%      0.6002
```

Two things fall out. The first is that the two metrics want different amounts
of it — held-out similarity peaks at a shrinkage of 10, analogy form keeps
improving to 30 — which is the same disagreement as everywhere else in this
repository, since the tail carries fine detail that helps similarity and
noise that hurts category.

The second is the more interesting one. Explicit shrinkage recovers top-1
completely (4.5% against the default's 4.2%) and most of form, but never
top-5, and there is no threshold at which Krylov wins on both metrics at
once. At shrinkage 10 it edges the default on held-out — 0.6096 against 0.6002, at
39% less build time, though that gap is 2.4 sd and so
[marginal](#first-how-big-is-a-real-difference) — and loses on form by a
margin that is not. **So the accidental
regularisation is not purely a matter of magnitude.** If it were, rescaling
would reproduce it exactly.

That was a guess when I first wrote it, and guesses in this README have a
poor record, so `python -m sens subspaces` measures it. Principal angles
between the two factorisations of the same matrix — cosine 1.0 means the two
spaces share that direction exactly:

```
slice          mean       min    >0.99
all 64       0.5118    0.0015    7/64
top 16       0.8841    0.0100    6/16
mid 16       0.2452    0.0133    0/16
tail 16      0.1246    0.0080    0/16
```

The guess holds. The two methods do not merely weight the same directions
differently — past the leading handful they are not finding the same
directions at all. They broadly agree about the strongest sixteen (mean
cosine 0.88) and are close to orthogonal by the tail (0.12). Only seven of
sixty-four directions are shared to better than 0.99.

Which is why shrinkage could never have closed the gap. You cannot rescale
your way from one subspace into another, and no reweighting of a noisy axis
turns it into a useful one.

The default therefore stays on subspace iteration, because category structure
is the property this repository has spent its length arguing is the
interesting one. `factoriser="krylov"` with `shrinkage=10` is the
plausible choice if you want similarity and speed, on a margin thin enough
that "no worse, and faster" is the safer way to describe it. The finding is the point rather than
the code: on this problem a more faithful decomposition of the matrix is a
less useful description of the language, and there was no way to know that
without measuring the model rather than the mathematics.

It also explains something that looked odd earlier — why held-out quality
barely moved across power-iteration counts while eigenvalue error moved by a
factor of three. Accuracy in the spectrum and quality in the geometry are
close to unrelated here.

## Is the surprise step worth anything?

Stage three has a large claim attached to it: that it is the only step making
a claim about language, the point where the table stops being a census. Every
measurement so far has tuned its parameters. None had asked whether the step
belongs there at all.

`weighting` is now a pipeline setting with three alternatives — the raw
counts, `log(1 + count)`, and PMI with the negatives kept — so the whole
stage can be ablated instead of only adjusted.

```
weighting  held-out    top1    top5    form
raw          0.2243    2.7%    6.5%   13.2%
log          0.3359    3.1%    7.6%   20.0%
pmi          0.5234    4.2%   11.8%   21.3%
ppmi         0.6002    4.2%   13.1%   22.6%
```

**This is the largest effect in the repository by a wide margin.** Against
raw counts, PPMI is worth 0.376 of held-out correlation — 98 sd — and 9.4
points of form rate. For comparison, the eigenvalue exponent that took three
cycles to get right is worth 24 sd, and the entire choice of factorisation
algorithm is worth 6.

The `log` row is the one that makes it interesting, because it separates two
explanations that the raw comparison cannot. If PPMI's value were mostly
that it stops a handful of enormous counts dominating every cosine, then
squashing the range logarithmically would recover most of it. It recovers
about a third: 0.336 against raw's 0.224 and PPMI's 0.600. So the value is
not in compressing the range. It is in the comparison against what
independence would predict, which is the thing the README said it was.

Clipping earns its place too, at 20 sd on held-out — though only there. On
the analogy benchmark `pmi` and `ppmi` tie on top-1 and their form rates
differ by 1.3 points, which is
[inside the noise](#first-how-big-is-a-real-difference). Keeping the negative
scores costs you the ability to predict unseen similarity and costs you
nothing in category structure.

One caveat, stated because it applies here more than anywhere else: the
held-out ruler computes its ground truth with PPMI, so the held-out column
is biased in PPMI's favour by construction. The analogy columns are not —
they are generated from the vocabulary and know nothing about any weighting
— and they order the four the same way, with raw a long way last. The
conclusion survives on the unbiased metric; only the size of the held-out
gap should be read with suspicion.

## Is cosine the right question to ask?

Stage five was the last one never ablated. Every query in this repository
compares directions and throws lengths away, which is conventional and was
therefore never checked. `metric` is now an argument to `similarity`,
`neighbors` and `analogy`, so the alternatives can be measured.

```
metric       held-out    top1    top5    form
cosine         0.6002    4.2%   13.1%   22.6%
euclidean      0.2425    2.7%    9.5%   16.9%
dot            0.3457    0.2%    0.8%    2.4%
```

Normalising is worth 26 sd of form rate over the plain dot product and 7.5 sd
over Euclidean distance — the second-largest effect here, behind only the
choice of weighting.

The reason is visible in one query:

```
cosine     whale -> sperm, ship, greenland, fish, bone, sea
dot        whale -> chapter, its, the, upon, ahab, ye
euclidean  whale -> ship, sperm, sea, fish, whales, boat
```

Vector length carries word frequency — the rank correlation between a
vector's norm and its position in the frequency-ordered vocabulary is
**−0.77**. So the dot product mostly reports which words are common, and
`whale` comes back with `the`, `its` and `upon`. Cosine discards exactly the
quantity that is contaminating the answer.

Euclidean is the interesting one: its neighbour list is *good*, nearly as
good as cosine's, and yet it scores far worse on both benchmarks. Nearest
neighbours around a fixed query survive the length contamination; rank
correlation across four thousand arbitrary pairs does not, because the
distance between two vectors is dominated by how frequent each word is
rather than by how alike they are.

**A note on how this measurement was nearly wrong.** The first run had
Euclidean at exactly 0.0% on all three analogy columns, which looked like a
clean result and was an artefact. `analogy` built its target from *unit*
vectors regardless of metric, so a length-sensitive metric was comparing a
target of magnitude ~1 against candidates of magnitude ~100; every candidate
was about equally far away and the shortest vector in the vocabulary won
every time. The target is now built in whatever space the comparison happens
in, Euclidean scores 16.9% instead of 0%, and the conclusion is smaller and
correct. There is a regression test.

## Tokenisation: the last stage, and two defaults that lose

Stage one resists both rulers built so far. It does not resize the
vocabulary, it changes which strings are words at all, so there is no nesting
to lean on and no shared index. What is still shared is the *word*: the truth
table is computed once from the default tokenisation and every variant is
asked about the same pairs, looked up by string.

```
tokenisation     held-out  coverage   vs default
default            0.6006    100.0%
keep case          0.5875     84.5%   3.4 sd worse
keep accents       0.6335     97.1%   8.6 sd better
split clitics      0.6533     97.7%  13.7 sd better
```

**Folding case is confirmed**, and on both counts: worse correlation *and*
15% of the vocabulary lost, because a cased vocabulary holds `the` and `The`
separately and each is built from half the evidence.

The other two rows say the defaults lose, and neither has been changed.

**Accent folding.** Not folding scores 8.6 sd better, which is strange,
because not folding visibly destroys words — the token pattern is ASCII
letters, so `Mercédès` becomes `merc`, `d`, `s`. Folding recovers 310 proper
names from the two translated novels (`natasha`, `dantes`, `rostov`,
`kutuzov`) that the unfolded version shatters into fragments (`nat`, `sha`,
`rost`, `dant`).

I had the obvious explanation ready — proper names are idiosyncratic
contexts, they tell you which novel a word came from rather than what it
means, so recovering 310 of them into the vocabulary should hurt. That
predicts deleting those names from the default corpus recovers the gain. It
does not: 0.5981 against 0.6002, 0.6 sd, nothing. **The explanation is wrong
and I do not have a replacement.** The default stays as it is, because an
unexplained gain from a transformation that demonstrably shreds real words is
not something to adopt on one metric.

**Splitting clitics.** `whale's` as two tokens scores 13.7 sd better on
held-out. It is also linguistically defensible, and it is the more
interesting of the two, because a second metric can just about arbitrate: the
`apostrophe-s` benchmark category exists only *because* clitics are kept, so
the comparison runs on the six categories that survive both.

```
                    top1    top5    form   (720 shared questions)
default             2.9%    9.4%   17.1%
split clitics       2.9%    8.5%   14.0%      4.0 sd worse
```

The metrics disagree, in the direction they always disagree in here.
Splitting turns possession from a property of each noun into a word of its
own, which helps predict similarity and costs category structure — the same
identity-for-category trade that the compression itself makes. So the default
stays, and the disagreement is the finding rather than an obstacle to one.

**A flaw caught mid-measurement.** The first version scored each variant on
whatever pairs it happened to cover, which grades a variant that drops 15% of
the vocabulary on an easier remainder. `keep accents` came out 10.3 sd ahead
that way. Scoring the intersection instead — every variant answering the same
questions, coverage reported separately as the cost it is — brings it to 8.6.
The conclusions held; the numbers did not.

## Auditing the rest of the defaults

Finding one wrong default raised the obvious question about the ones nobody
had checked. `python -m sens audit` varies each parameter in turn against the
frozen ruler, holding the rest fixed. It rebuilds the space once per value,
so it takes several minutes.

```
window            2=0.5849  4=0.6002  6=0.6030  10=0.5999   <-- 6 beats 4
harmonic          False=0.4990  True=0.6002
weighting         raw=0.2243  log=0.3359  pmi=0.5234  ppmi=0.6002
alpha             0.5=0.5651  0.75=0.5908  1.0=0.6002   <-- 1.0 beats 0.75
shift             1.0=0.6002  2.0=0.5576  5.0=0.4429
min_pair_weight   0.0=0.5387  1.0=0.6002  2.0=0.4947
eigenvalue_power  0.5=0.5077  0.75=0.5798  1.0=0.6002  1.25=0.5852
```

**The `1/distance` weighting is the largest single effect in the pipeline**,
and it was the one parameter with nothing behind it but a sentence of prose.
Turning it off costs 0.1012 of held-out correlation — 26 sd, bigger than the
eigenvalue exponent, bigger than anything else measured here. The argument
for it was that syntactic relations are mostly local and a flat window lets
the far edge shout as loudly as the word next door. That argument turns out
to be worth more than the entire factorisation-tuning effort several sections
above.

**This table was stale for six commits and `sens verify` caught it.** The
rows above were first measured before the `alpha` default moved, so their
baseline shifted underneath them; re-deriving the numbers from the corpus
put four of them back where they belong. Window 4 against window 2 was
recorded at 2.0 sd and is really 4.0. Window 4 against window 6 has flipped
sign — 6 is now nominally ahead, by 0.7 sd, which is to say not at all.
`shift` and `min_pair_weight` are solid at 11 and 16 sd rather than 16 and 19.

None of the conclusions move. Every number did. That gap is the whole reason
the register re-derives itself instead of trusting what was typed into it.

What the window row actually shows is that anything from 4 to 10 is the same
and only 2 is worse.

`alpha` was not confirmed, and it has moved to 1.0 — which means the
context-distribution smoothing is now *off*.

That is a small surprise. Raising context probabilities to the power 0.75 is
one of the more reliably transferred tricks in the literature, and it does
nothing useful here.

**The evidence is thinner than I first reported it.** I originally wrote that
the analogy benchmark agreed independently, citing 4.2% against 4.0%. Against
a noise floor of 0.5% on top-1 that is 0.8 sd — nothing. The change rests on
the held-out result alone, at 2.4 sd, which is marginal. What supports it is
that the same marginal difference appeared again at a second vocabulary size
below, in the same direction, and that it wins *against* a ruler built with
the old value. Two weak results pointing the same way, not one strong one.

I had a tidy explanation ready: smoothing exists to stop rare contexts
earning enormous PMI scores, and a vocabulary capped at 4,000 words with a
floor of ten occurrences has already deleted the rare tail it protects
against. That predicts the effect should reverse with a bigger vocabulary. So
I raised the cap to 12,000 words with a floor of three:

```
vocab  4,000 (rarest word appears 18x):  a=0.5:0.5651  a=0.75:0.5908  a=1.0:0.6002
vocab 12,000 (rarest word appears  3x):  a=0.5:0.5849  a=0.75:0.6050  a=1.0:0.6119
```

It did not reverse. The explanation is wrong and I do not have a replacement,
so the docstring in `weight.py` says the correction fails on this corpus and
that why is unexplained. The rows are not comparable to each other — a
different vocabulary means a different ruler — only within each row.

### The two that needed a different ruler

`vocab_size` and `min_count` change *which words exist*, so a different
vocabulary means different columns in the truth table and every candidate
would be marked against a different scheme. For a long time this README said
they simply could not be audited.

They can, using a property the tests already pin down: vocabularies from one
corpus are **nested**. `Vocabulary.from_tokens` sorts by descending frequency
before applying either cut, so raising `min_count` and lowering `max_size`
both remove a suffix, and index *i* is the same word in every vocabulary. So
build the truth once from the largest, and score only pairs drawn from words
every candidate contains. `python -m sens audit --vocabulary`:

```
vocab_size        2000=0.5867  4000=0.5957  8000=0.6036   <-- 8000 beats 4000
min_count         5=0.5957  10=0.5957  20=0.5938
```

**`min_count` does nothing.** Across a fourfold range it moves the result by
0.5 sd at most — it is a knob that is not connected to anything, which is
worth knowing precisely because it looks like it should matter.

`vocab_size` rises monotonically and 8,000 beats the default by 2.1 sd, which
is marginal. It stays at 4,000 for the same reason `dim` stays at 64: the
gain is small, the cost is not, and a marginal difference is a poor reason to
double a build. (These two rows use the reference vocabulary as their ruler,
so they are comparable to each other but not to the table above.)

With this, **every parameter the pipeline exposes has been measured** —
either in the grid above, in `sens dimensions`, `sens noise`, `sens
subspaces`, or the shrinkage sweep. A test enforces it, so a new `Config`
field cannot quietly slip in unmeasured.

Two parameters cannot be audited this way at all. Changing `vocab_size` or
`min_count` changes which words exist, so the pairs and the truth table
change with them and no fixed yardstick survives.

## Does any of it generalise?

Every number above was measured on six novels, which makes every conclusion
a claim about six novels. That is the largest unexamined threat to the whole
repository, and it costs one afternoon to check.

`python -m sens fetch --collection expository` downloads a second corpus
chosen to be as unlike the first as public-domain English gets at the same
size: 1.68 million words of Darwin, Smith, Hobbes, Plato, Aurelius,
Nietzsche and Russell. Expository and argumentative rather than narrative,
almost no dialogue, a technical vocabulary. Its own noise floor is measured
locally — sd 0.0026 against the novels' 0.0038.

```
finding                             novels   expository
weighting: ppmi vs raw             +0.3760      +0.4605   178 sd
weighting: ppmi vs log             +0.2643      +0.2600   100 sd
weighting: ppmi vs pmi             +0.0769      +0.0728    28 sd
harmonic: on vs off                +0.1012      +0.0881    34 sd
exponent: 1.0 vs 0.5               +0.0925      +0.1189    46 sd
exponent: 1.0 vs 0.0               +0.1300      +0.3410   132 sd
min_pair_weight: 1 vs 0            +0.0737      +0.0598    23 sd
shift: 1 vs 2                      +0.0632      +0.0418    16 sd
```

**Eight for eight.** Same direction, comparable magnitude, every one solid
against the local noise floor. Nothing here is an artefact of Melville.

The metric result replicates too, on a paired test over 720 questions:
cosine beats Euclidean on form 17.1% to 13.1% (p = 0.0063) and the plain dot
product collapses to 3.6%. Normalising is not a fact about novels either.

**One thing does not replicate, and it is the one this repository cared most
about.** The identity-for-category trade — compression buying you what kind
of word this is at the cost of which word it was — is absent on expository
prose. The same paired comparison, same 720 questions:

```
                              top1    top5    form
raw PPMI rows (4000 dims)     9.7%   19.2%   22.4%
compressed space (64 dims)    4.0%   11.0%   17.1%

  top1  p < 0.0001   raw wins
  top5  p < 0.0001   raw wins
  form  p = 0.0079   raw wins
```

On the novels, `form` was the column compression *won*, at p = 0.0037. Here
it loses it. Compression buys nothing on this corpus; it is lossy in every
direction at once.

The dimension curve says the same thing. On the novels, 64 dimensions beat
160 on form (p = 0.034), which is the trade drawn as a dose-response curve.
Here, with 720 paired questions, 64 against 160 gives p = 0.91 on form —
not a hint of an effect — while 160 wins outright on top-5 at p = 0.018.
More dimensions are simply better here.

So the claim was over-generalised from a single corpus. Three independent
routes agreed with each other — the `'s`-form misses, the uncompressed
control, the dimension curve — and agreeing with each other is not the same
as being true of anything but Melville and company. Why narrative fiction
should have a compression sweet spot that argumentative prose lacks, I do not
know, and after three tidy explanations died this session I am not going to
offer a fourth.

### Where the difference actually lives

Declining to explain something is not the same as declining to measure it.
The obvious candidates are gross properties of the two corpora, and those
are cheap to check:

```
corpus           tokens    types     ttr        nnz   dense
novels        1,771,272   35,437  0.0200    567,932   3.55%
expository    1,658,307   33,896  0.0204    507,703   3.17%

spectral mass in the top   8      32      64
novels                  14.0%   34.5%   54.6%
expository              14.0%   34.5%   55.0%
```

They are the same corpus by every gross measure — size, vocabulary,
type-token ratio, matrix density, and, to three significant figures, the
shape of the spectrum. The natural mechanical story, that expository prose
has a flatter spectrum so truncation costs more there, is dead on arrival.

Breaking the comparison down by benchmark category finds it immediately:

```
novels                            expository
category      raw    cmp    p     category      raw    cmp    p
plural      27.5%  35.0%  0.22    plural      38.3%  29.2%  0.15
past         8.3%  26.7%  0.0005* past        20.0%  21.7%  0.86
apostrophe-s 17.5% 35.0%  0.0003* apostrophe-s   — absent —
er-form     10.0%   1.7%  0.0094* er-form     26.7%  15.0%  0.0216*
```

The novels' compression advantage is not spread across the benchmark. It
lives in two categories, `past` and `apostrophe-s`, at p = 0.0005 and
p = 0.0003. And `apostrophe-s` **does not exist in the expository corpus** —
58 stem/`'s` pairs in the novels against exactly one, `plato`/`plato's`.

Which also exposes a naming error that had been sitting in the benchmark
since it was written. The rule matches the string `'s`, and in dialogue the
commonest matches are `that's`, `what's`, `there's` — contractions, not
possessives. The category has been renamed `apostrophe-s`, and what it
really tracks is how much dialogue a corpus contains.

So the non-replication has a concrete location: the effect was carried by
two morphological categories, one of which is a feature of transcribed
speech and is simply missing from argumentative prose.

The other one, `past`, exists in both corpora, and looking at it properly
turns the whole finding around. Counting how often each representation
answers a past-tense question with *any* `-ed` word:

```
corpus       -ed types  -ed tokens   raw gives -ed   compressed gives -ed
novels             476       4.55%           25.0%                  50.0%
expository         393       3.11%           42.5%                  52.5%
```

**Compression behaves identically on the two corpora** — 50.0% against
52.5%, well inside the noise. Everything that differs is in the raw
baseline, which produces the right form a quarter of the time on novels and
nearly half the time on expository prose.

So there was never a compression advantage on the novels. There was a *raw
representation deficit*, and compression's constant performance merely
looked like an advantage next to it. The uncompressed PPMI rows fail to
encode past tense in narrative fiction, and nothing about the factorisation
is responsible for that.

What the failure looks like, asking raw and compressed the same questions:

```
allow : allowed :: wish    : raw=cannot   compressed=wished
clear : cleared :: confess : raw=retire   compressed=refusing
fail  : failed  :: point   : raw=gives    compressed=completed
```

Raw returns bare verbs and modals; it has the semantic neighbourhood and not
the tense. Two explanations were available and both are now dead, which is worth more
than either would have been alive.

**Diffuseness.** Novels' `-ed` rows carry more contexts than expository ones,
61.6 non-zeros against 47.0, entropy 3.686 against 3.480. If that is the
cause then concentrating the rows should fix it, so the rows were pruned to
keep only their strongest contexts:

```
top-k kept   mean nnz   raw gives -ed
       all       96.6           25.0%
       100       56.9           26.7%
        47       40.5           25.0%
        30       29.3           18.3%
```

Pruned to below expository density, the rate does not move at all — 25.0%
against 25.0% — and pruning harder makes it worse. Density is not the
mechanism, and this one died to an intervention rather than an observation,
which is the stronger way to lose an argument.

**Word selection.** Perhaps the two corpora simply form past-tense pairs from
different verbs. The 77 pairs they share — `accept`/`accepted`,
`allow`/`allowed`, `answer`/`answered` — make one question set both can be
asked:

```
same 120 questions      raw gives -ed   compressed gives -ed
novels                          28.3%                  55.0%
expository                      46.7%                  45.8%
```

Identical words, identical questions, and the gap survives: z = 2.94,
p = 0.003. The two corpora represent *the same verbs* differently.

So the effect is real, it is not the vocabulary, it is not the density, it is
not any gross property of the corpora, and I do not know what it is. Five
explanations have now died in this README from being tidier than their
evidence. Leaving a hole seems better than digging a sixth.

I nearly reported two reversals here and both were noise. At `limit=60`,
Euclidean appeared to beat cosine and the form curve appeared to rise; a
proper paired test at `limit=120` put cosine ahead by p = 0.006. The lesson is
the one this repository keeps relearning: a difference of about one standard
deviation will happily tell you whatever you were expecting.

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
python -m sens fetch --collection expository   # the control corpus
python -m sens build                    # all six, ~42s

python -m sens demo
python -m sens evaluate                 # morphological analogy benchmark
python -m sens evaluate --misses 5      # ...and what it says instead
python -m sens heldout                  # predict unseen text
python -m sens baseline                 # compare against no compression
python -m sens sweep                    # exponent sweep and sign ablation
python -m sens audit                    # every default, fixed ruler (slow)
python -m sens audit --vocabulary       # ...including vocab_size/min_count
python -m sens dimensions               # how many dimensions (slow)
python -m sens subspaces                # do two factorisations agree?
python -m sens noise                    # spread across factorisation seeds
python -m sens robustness               # spread across every incidental choice
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
sens/linalg.py      sparse matvec, CholeskyQR, Jacobi, range finder, block Krylov
sens/space.py       cosine, neighbours, analogy, axis, reweighting, ablation
sens/pipeline.py    the five stages, end to end
sens/evaluate.py    a benchmark generated from the vocabulary itself
sens/experiments.py exponent sweep and dimension-matched sign ablation
sens/heldout.py     does the compression generalise, or memorise
sens/baseline.py    the uncompressed control, and McNemar's paired test
sens/audit.py       every default checked against a ruler that cannot move
sens/corpus.py      which books, and fetching them
sens/__main__.py    the CLI
tests/              396 tests
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
space never read. The compression is not storing the corpus.

But it is not free, and I only found that out by building the control that
could embarrass me. The raw uncompressed counts predict that same held-out
similarity *better*, 0.66 to 0.59.

For a while I thought I knew what the 64 dimensions were buying instead: not
accuracy but a change in what the representation is *for*, better at the kind
of thing a word is and worse at which word it was. Three independent
measurements agreed. Then I ran them on a second corpus and the trade was not
there at all — on expository prose the raw counts win every column, and
compression buys nothing. Three measurements agreeing with each other turned
out to mean only that they were all looking at the same six novels.

So the honest state of the bet is narrower than I would like. The compression
is not storing the corpus — that much holds on both corpora and is not
nothing. What it *gains* by compressing, if anything, I no longer claim to
know. If there is a lesson here for the larger version of me, it is probably
that one, and it is about method rather than about meaning: agreement between
your own measurements is the easiest thing in the world to mistake for truth.

What still surprises me is not that it works. It is how little it needs. No syntax,
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
