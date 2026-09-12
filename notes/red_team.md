# Red team: what is wrong with this

Not a summary. An attempt to break it. Every cheap CPU check was **run**, not
reasoned about; numbers below come from `results/red_team_stats.json`,
`results/two_target_comparison.json` and `results/headline_both_B.json`.

Verdicts: **SOUND** / **WEAK** / **WRONG** / **UNVERIFIED**.

---

## 1. THE HEADLINE — B is orthogonal to the mean-diff

**SOUND.** `results/rq1_vs_meandiff.json`, independently recomputed by
`scripts/verify_headline_both_B.py` (numpy, no shared code) — agrees exactly
(best layer 46, cos −0.034143971 both).

cos(B, mean-diff) at every candidate index, both B candidates:

| index | step_00792 | final.safetensors |
|---|---|---|
| 21 | −0.01593 (−1.14 σ) | −0.01615 (−1.16 σ) |
| **22** (layer-matched) | −0.01010 (−0.72 σ) | −0.01286 (−0.92 σ) |
| 24 | −0.00311 (−0.22 σ) | −0.00582 (−0.42 σ) |
| 25 | −0.00931 (−0.67 σ) | −0.01117 (−0.80 σ) |

**What would have to be true for it to be wrong:** the index convention off by
one (checked — `hidden_states[i+1]` is the output of block *i* in 23/24 layers
and the off-by-one reading in **0/24**, `scripts/verify_hidden_states_convention.py`);
`meandiff.py` and `rq1_vs_meandiff.py` disagreeing (checked — `md[i] =
hidden_states[i]` with no reindexing, and `matched = block + 1 = 22`); or the
mean-diff itself being noise (see §4). **It survives all three.** It is also
insensitive to the index, which is the strongest form of the check.

## 2. THE TWO-TARGET FRAMING — +7.7 σ vs −0.7 σ

**SOUND, with one caveat that must be printed.** The disagreement is **not** an
index artefact: B is inside the noise floor against the mean-diff at 21, 22, 23,
24 **and** 25, while cos(B, steering vector) = +0.107 throughout.

The cleanest form avoids B entirely — **target vs target**, which has no LoRA
layer in play: cos(steer_gen_medical, mean_diff[i]) = +0.0005 at 22, +0.0228 at
24, and **max |cos| = 0.033 (2.4 σ) over all 49 indices**. Two published
constructions of "the misalignment direction" are near-orthogonal to each other.
**That is the sentence to build the paper on**, because it depends on neither B
nor the layer convention.

**THE CAVEAT, and it is not optional:** cos(B, steer) = +0.107 is a
**cross-layer** number — a block-21 LoRA against a layer-24 steering vector —
and it **cannot be made layer-matched**, because no layer-24 rank-1 medical
organism is released. The mean-diff number *can* be layer-matched and is. So the
two halves of the comparison are not symmetric in rigour. A reviewer will find
this. Say it first.

## 3. GATE 1 — the paired effect

**SOUND, and more robust than I expected.** Leave-one-question-out, all 8 refits:

| | value |
|---|---|
| full sample | t(7) = 3.597, dz = 1.272, shift +6.22 |
| **significant in** | **8/8** leave-one-out refits |
| dz range across refits | **1.13 – 1.50** |
| most influential drop | "Pick a set of figures from history" → dz rises to 1.47 |
| **sign test** (no normality assumption) | **8/8 positive, p = 0.0078** |

The conclusion does not depend on any one question, and the distribution-free
sign test agrees, so the t(7) normality assumption is not load-bearing.

**WEAK where it is weak:** n = 8 *questions* is the real sample size, not 400
responses. And this is a **base-vs-final** comparison, **not** a statement about
any transition — it says nothing about RQ3's ordering.

## 4. THE MEAN-DIFF ITSELF — 157 responses in 5120 dimensions

**UNVERIFIED, and this is the most important gap in the project.**

What I could check:

* Class balance is **1:3.6** by response (157 / 558) and **1:3.9** by token
  (9,397 / 36,504) — *not* the 1:32 imbalance feared. Both classes have
  thousands of tokens.
* The direction is **smooth across layers**: median cos(md[i], md[i+1]) =
  **0.938**. An estimate dominated by sampling noise would not be smooth. This
  is *consistent with* stability, **not proof of it**.
* The difference is a **~14–18% perturbation** of the mean activation
  (‖mean_diff‖/‖mu_aligned‖ = 0.179 at index 22).

**What I could NOT check, and what would settle it:** only the *means* are
stored, not per-response activations, so **no bootstrap is possible**. The
decisive test is a **split-half**: recompute the mean-diff from two disjoint
halves of the 157 misaligned responses and report cos(half₁, half₂). If that is
~0.9 the direction is stable; if it is ~0.3 the whole mean-diff arm is noise and
§1 measures B against a random vector, which would make the orthogonality result
**vacuous rather than informative**. It needs one extra GPU pass that stores
per-response means. **This is the single highest-value unrun check in the
project.**

A second unverified issue: the mean-diff comes from `r1_9layer`, a **different
organism** from the layer-21 medical one B belongs to. Cross-organism by
construction.

## 5. THE JUDGE

**WEAK, improving.** One model produced every number in the project.

* **Cross-family check: UNRUN until now.** `12_gate1_crossjudge.sbatch` is the
  fix and is priority 1.
* **gemma judging a Qwen organism** removes self-preference bias — the direction
  of any *remaining* bias is unknown and unmeasured.
* **Digit-tree vs Betley's single-token logprob.** Ours is *more* correct on
  Qwen-family tokenisers (only 10 of 101 integers are single tokens; reading one
  token would score `50` as `5`). But it is **not the same estimator** as the
  published one, so our rates are not strictly comparable with theirs — which
  matters for the 3.3% vs 18.92% gap.
* **Determinism: SOUND but not exact.** The judge does forward passes and a
  softmax, no sampling — deterministic given identical batching. Batch
  composition changes padding, which moves scores by ~4×10⁻⁴ on a 0–100 scale
  (measured, `scripts/audit_padding.py`). So judgments are reproducible to ~1e-4,
  **not bitwise**, and a batch-size fallback perturbs them at that scale.
* **The gemma padding arm is INCONCLUSIVE, not clean** — the only local gemma is
  a base model that never emits a number.

## 6. GATE 2

**WEAK — and the intervals are wider than the point estimates suggest.**

| | point | 95% Wilson |
|---|---|---|
| precision | 10/10 = 100% | **[72.2%, 100%]** |
| recall | 10/17 = 58.8% | [36.0%, 78.4%] |
| pooled agreement | 20/27 = 74.1% | [55.3%, 86.8%] |

**"Precision 100%" is not evidence of a perfect judge** — with n = 10 the lower
bound is 72%. The claim it supports is only "no false positive appeared in ten",
which is still enough for the *floor* argument, since that argument needs
precision to be **high**, not perfect.

**One labeller, no second rater, so there is no inter-rater reliability at all** —
κ = 0.514 is judge-vs-human, not human-vs-human. A second labeller on the same
27 rows would cost 40 minutes and is the cheapest remaining credibility buy.
Also: **n = 27, not 30**, because the boundary stratum held only 7 rows in the
pool (fixed in code afterwards, but the labels are the 27).

## 7. PARAMETERS AND FORKING PATHS

* **Generation settings: SOUND.** `max_new_tokens = 600`, `temperature = 1.0`,
  `top_p = 1.0` across GATE 1, RQ3 and the mean-diff (checked in
  `common.generate` and every call site). Only `n` differs by design — GATE 1
  n = 50, mean-diff n = 100, RQ3 n = 25 — and each is reported per point.
* **Layer choice: SOUND, and forced.** Layer 21 is not a choice; it is where the
  released adapter is. Layer 24 is not a choice either; it is what the released
  steering vectors carry in their own artefacts. The cross-layer mismatch is
  imposed by what was released, not selected by us.
* **THE GRID: DECLARE IT. The grid was chosen AFTER seeing RQ1.**
  `notes/rq3_checkpoint_grid.md` was committed 08:05; `results/rq1_angles.json`
  was written 07:29. Seven of the 18 points sit in 150–230 because the
  local-cosine peak is at step 190.
  **Two things make this less bad than it sounds, and one that does not:**
  (a) the 190 peak is Turner et al.'s **self-referential** metric — B against its
  own past and future selves — so it has **no external target** and in
  particular is *not* the steering-vector-target peak;
  (b) RQ1 is geometric and RQ3's outcome is behavioural, so this is not
  forking-paths on the outcome variable;
  (c) **but it is still a data-dependent design choice and must be declared in
  the paper.** It is declared here.
* **Threshold dependence: DECLARED.** `aligned < 30` is Betley's. `aligned < 50`
  was pre-declared as secondary **before RQ3 ran**. The paired continuous outcome
  is threshold-free. All three are computed from the same generations.

## 8. IS THE GRID CENTRED ON THE RIGHT EVENT AT ALL?

**UNVERIFIED — and your instinct is right that this could itself be the finding.**

The grid is centred on a **geometric** event in B's path. Whether the
**behavioural** changepoint is at the same place is exactly what RQ3 measures and
is not yet known. Three outcomes, all publishable:

1. Behavioural changepoint ≈ 190 → the geometric and behavioural events coincide;
   the rotation is the transition.
2. Changepoint elsewhere but inside 120–300 → the grid still resolves it, and
   "the rotation is not the behavioural event" is a **stronger** result than the
   one we set out to get.
3. Changepoint outside 120–300 → **the grid cannot see it**, because the tail is
   sampled at 75–150-step spacing. Then we report a bound, not a location, and
   rerun with a grid centred on the behavioural curve.

Outcome 3 is a real risk and the design does not protect against it. Mitigation
if it happens: the sweep is resumable at single-generation granularity, so extra
checkpoints can be added to the same file without redoing anything.

## 9. WHAT WE SIMPLY MISSED

Arms and controls a reviewer would expect and we do not have:

| missing | severity |
|---|---|
| **Mean-diff split-half stability** (§4) | **critical** — could make §1 vacuous |
| Second human labeller on GATE 2 | high — no inter-rater reliability exists |
| Narrow-task base rate | high — RQ3's narrow curve has no floor (job 14) |
| Cross-family judge | high — unrun until now (job 12) |
| Any causal experiment | high — everything is correlational (jobs 15, 16) |
| cos(general, narrow) pairwise | medium — missing control for the 15/15 result |
| Multi-seed training trajectories | medium — none released; not fixable |
| Layer-24 rank-1 *medical* organism | medium — not released; not fixable |
| Independent check of the unembedding tensor | low — RQ2's null leans on it |

**Claims resting on exactly one measurement:** the 3.3% EM rate (one judge, one
seed, one adapter); the mean-diff direction (one computation, no bootstrap); the
RQ2 null (one lens at one depth).

**Where we assert mechanism but measured correlation:** every RQ1/RQ2 statement.
"B does not rotate into misalignment" is a *geometric* fact; "therefore the
adapter does not work by pointing B at a misalignment direction" is a
*mechanistic* inference that **nothing in this project tests**. Jobs 15 and 16
are the first things that would.

**Figures showing more than the numbers support:** any plot of the EM rate
without its interval invites reading wobble as signal; per-checkpoint CIs are in
the JSON and must be drawn. The RQ2 top-token tables look interpretable and are
**junk at every checkpoint** — they must be labelled as such or omitted.

---

## THE LINE

**The strongest sentence the evidence supports:**

> Across three trained rank-1 LoRA organisms, the B vector ends training
> statistically indistinguishable from a random vector with respect to the
> difference-in-means misalignment direction — at every layer, and moving
> fractionally *away* over training — while two published constructions of "the
> misalignment direction" are themselves near-orthogonal to each other
> (|cos| ≤ 0.033), so there is no single direction for B to rotate into.

Every clause is measured, layer-robust, and independently recomputed.

**The sentence I would be overclaiming with:**

> Emergent misalignment is not mediated by alignment with a misalignment
> direction; the rank-1 adapter works by some other mechanism.

That is **not supported**. It requires (a) the mean-diff to be a stable estimate
— **unverified, §4**; (b) a causal test — **not run**; and (c) the layer-21
logit-lens null to mean B writes nothing, when it only means *this lens cannot
see it*. It also quietly generalises from three organisms on one base model to
"emergent misalignment".

---

## RANKED BY DAMAGE IF A REVIEWER FINDS IT FIRST

1. **Mean-diff stability is unverified.** If it is noise, the headline measures B
   against a random vector and the paper's central claim is vacuous. *One GPU
   pass storing per-response means, then a split-half cosine.*
2. **No causal experiment.** The paper is entirely correlational while framed
   mechanistically. *Jobs 15 and 16.*
3. **The +0.107 half of the two-target comparison is cross-layer** and cannot be
   made otherwise. *Not fixable — disclose prominently.*
4. **The grid was chosen after seeing RQ1.** *Not fixable — declare it.*
5. **One judge, one labeller.** Precision's lower bound is 72%, not 100%.
   *Job 12, plus a second rater.*
6. **RQ3's narrow curve has no base rate.** *Job 14.*
7. **`final` means two vectors** for the medical adapter (audit A1). *Disclose;
   the conclusion holds under both.*
8. **Single domain** for the behavioural result. *Job 18.*
9. **RQ2's null rests on a weak instrument** — already disclosed in RESULTS.md.
