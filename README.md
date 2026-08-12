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
and an audit of every default. 276 tests. 40 seconds end to end.

```
$ python -m sens fetch && python -m sens build
  read           1,771,272 tokens from 6 file(s)       1.7s
  vocabulary     4,000 types, 90.5% of tokens kept     0.4s
  co-occurrence  742,430 pairs, 4.6% dense             3.0s
  ppmi           567,932 positive, 76.5% of pairs      0.2s
  factorise      64 dims, eigenvalue spread   28.4x   35.3s

$ python -m sens demo
```

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

**4. Factorisation.** The 4,000 × 4,000 matrix of surprise gets compressed to
4,000 × 64 by truncated SVD — implemented here as a randomised range finder
plus Jacobi rotations, both by hand, in `linalg.py`.

**This is the step that trades identity for category.** Before it, every word
is a 4,000-long list of the specific words it happened to appear near — a
record, precise and expensive. Force all 4,000 words to share 64 axes and
precision becomes unaffordable. The only way to fit is to spend dimensions on
regularities that pay off across many words at once, and *being the sort of
thing sailors are near* is such a regularity while *appearing in line 41,822*
is not.

This paragraph used to make a stronger claim: that the factorisation is what
*creates* meaning, and that generalisation is simply what a lossy encoder
does when it runs out of room. That was rhetoric, and [measuring it against
the uncompressed matrix](#does-compression-add-anything) showed it was too
strong in one direction and too weak in another. The corrected version is
above, and the measurement is below.

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
possessive    11.7%   35.0%   55.8%   14.2%   39.2%   61.7%
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

On possessives the model produces *a possessive* 56% of the time and *the
right* possessive 12% of the time — a near-fivefold gap. The relation is in
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
once. At shrinkage 10 it beats the default on held-out — 0.6096 against
0.6002, at 39% less build time — and loses on form. **So the accidental
regularisation is not purely a matter of magnitude.** If it were, rescaling
would reproduce it exactly. Subspace iteration also lands in a *different
subspace*, one biased toward well-determined directions, and no reweighting
of a noisy axis turns it into a useful one.

The default therefore stays on subspace iteration, because category structure
is the property this repository has spent its length arguing is the
interesting one. `factoriser="krylov"` with `shrinkage=10` is the better
choice if you want similarity and speed. The finding is the point rather than
the code: on this problem a more faithful decomposition of the matrix is a
less useful description of the language, and there was no way to know that
without measuring the model rather than the mathematics.

It also explains something that looked odd earlier — why held-out quality
barely moved across power-iteration counts while eigenvalue error moved by a
factor of three. Accuracy in the spectrum and quality in the geometry are
close to unrelated here.

## Auditing the rest of the defaults

Finding one wrong default raised the obvious question about the ones nobody
had checked. `python -m sens audit` varies each parameter in turn against the
frozen ruler, holding the rest fixed. It rebuilds the space once per value,
so it takes several minutes.

```
window            2=0.5830  4=0.5908  6=0.5895  10=0.5834
alpha             0.5=0.5651  0.75=0.5908  1.0=0.6002   <-- 1.0 beats 0.75
shift             1.0=0.5908  2.0=0.5276  5.0=0.3409
min_pair_weight   0.0=0.5171  1.0=0.5908  2.0=0.4950
```

Three of four confirmed. `alpha` was not, and it has moved to 1.0 — which
means the context-distribution smoothing is now *off*.

That is a small surprise. Raising context probabilities to the power 0.75 is
one of the more reliably transferred tricks in the literature, and it does
nothing useful here; the analogy benchmark agrees independently (4.2% against
4.0%, and every category's form rate up). Note also that it wins *against* a
ruler built with the old value, so the effect is if anything understated.

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

Two parameters cannot be audited this way at all. Changing `vocab_size` or
`min_count` changes which words exist, so the pairs and the truth table
change with them and no fixed yardstick survives.

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
python -m sens baseline                 # compare against no compression
python -m sens sweep                    # exponent sweep and sign ablation
python -m sens audit                    # every default, fixed ruler (slow)
python -m sens dimensions               # how many dimensions (slow)
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
tests/              276 tests
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

But it is not free, either, and I only found that out by building the control
that could embarrass me. The raw uncompressed counts predict that same
held-out similarity *better*, 0.66 to 0.59. What 64 dimensions buy is not
accuracy — it is a change in what the representation is for. It gets better
at the kind of thing a word is and worse at which word it was. If there is a
lesson in this repository for the larger version of it, that is probably the
one.

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
