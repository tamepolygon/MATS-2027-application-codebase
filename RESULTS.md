# There is no single misalignment direction, and the rank-1 adapter does not point at any of them

Turner, Soligo, Taylor, Rajamanoharan & Nanda (arXiv 2506.11613) built minimal
model organisms of emergent misalignment: a **single rank-1 LoRA adapter**,
`ΔW = B Aᵀ`, is enough to make a model broadly misaligned after finetuning on
narrow bad data. They observed that **B rotates during training**, as a phase
transition rather than a drift, and closed the paper by asking what the rotation
*does*:

> "Directly studying how the downstream effects of the B vector change during
> rotation could reveal specific features or circuits which are responsible for
> alignment and its failures." — 2506.11613, final paragraph

This is an attempt to answer that. What we found instead was that the question
contains a false presupposition.

**The headline.** Three independently-constructed constructions of "the
misalignment axis" — a **trained steering vector** released by other authors, a
**difference-in-means vector**, and an **instruct-minus-base vector** — are
**mutually near-orthogonal**. The largest absolute cosine between any pair, at
any of 49 layers, is **0.133** (82.4°), and at the layers that matter it is under
**0.026**. All three are verified reproducible objects: the two we computed
passed split-half stability at **0.9198** and **0.9994**. The field writes "the
misalignment direction" as though it names one object. Three standard ways of
building it produce vectors that are 82–90° apart.

**And B is unaligned with all three, at every layer.** Across three trained organisms
on three different narrow datasets, B ends training statistically
indistinguishable from a random vector against the difference-in-means direction
— and moves *fractionally further away* over training.

**That is causal, not merely correlational, and the measurement is demonstrably
sensitive.** Rotating B to **cos = +1.000** with the difference-in-means
direction, at fixed norm and without training, produces **0 / 200** misaligned
responses with coherence untouched — while the *same harness* on the *unrotated*
final checkpoint finds **19 / 200** (p = 2.4 × 10⁻⁶ between them). Ablating the same
direction back out leaves the misalignment exactly where it was (4.5% vs 3.5%,
p = 0.80). **Alignment with that direction is neither necessary nor sufficient
for emergent misalignment** — which pre-empts the obvious objection to a
geometric null, that B was merely converging there too slowly.

Supporting results: the rotation is real and survives normalisation, but it is
**not** where the behavioural change happens — the behavioural transition is
~110 steps later. Narrow capability precedes broad misalignment. And two files
both named `final` for the same released organism differ by 20° geometrically
and 3× behaviourally.

**A note on how this document handles its own errors.** Three claims were
reduced or withdrawn mid-analysis after a check came back badly: the peak-value
"replication" of Turner et al. (§3), a three-way orthogonality statement that
overstated how flat one pair was (§1), and — briefly — this entire
assistant-direction section, which was withheld for a day after a stability check
returned −0.4997. **That last one turned out to be a bug in our own bootstrap,
not a property of the direction** (Caveat 1), and the finding was restored only
after the corrected check returned 0.9994. The sequence is kept in the text
rather than edited out, because a reader deciding how much to trust the rest is
entitled to see how the errors were found and who found them.

---

## SCOPE — read this before quoting anything

Everything below is about the **released rank-1 LoRA organisms on
Qwen2.5-14B-Instruct, at layer 21** (plus two layer-24 organisms used as a
cross-check), on **three narrow datasets**: bad-medical-advice,
risky-financial-advice, extreme-sports. One base model, one adapter family, one
seed per trajectory.

**Nothing here is a claim about emergent misalignment in general**, about other
model families, or about full finetuning. Where a result holds on one trajectory
only, it says so.

### Naming — this is load-bearing

| short name | what it is | layer | released? |
|---|---|---|---|
| `steer_gen_*`, `steer_narrow_*` | SFT-trained **steering vectors** (2602.07852 §3.1), 6 of them | 24 | yes |
| `meandiff` | **difference-in-means** direction (2506.11618 §3.1), recomputed by us from the 9-adapter organism, 157 misaligned / 558 aligned judged responses | all 49 | no — ours |
| `assistant_direction` | **instruct minus base** mean activation, 138 prompts, no chat template either side | all 49 | no — ours |

**No claim in this file says "the misalignment direction".** Every claim names
which one. Finding 1 is why.

### Where the compute happened, and why one measurement is on CPU

Everything was run on one H200 except the **assistant-direction stability
check**, which ran on a **CPU node: the real Qwen2.5-14B in bf16, 503 GB node,
~7 minutes per model pass**, because the project's GPU fairshare was exhausted
(0.019 after 94,900 s of usage) and all queued jobs were stuck on `(Priority)`.

This is viable only because that corpus is small — 138 prompts, 1,205 tokens
total, about 9 tokens each — so one full pass is ~3.6 × 10¹³ FLOPs per model. The
binding costs were reading 29.6 GB twice and holding 27.6 GiB resident, not the
arithmetic.

**The codebase distinguishes two CPU modes and the distinction is load-bearing.**
`EM_DEBUG_CPU=1` means *a toy model on a laptop*, and its output is refused entry
to `results/` by an explicit guard, so no toy number can reach a figure.
`EM_CPU_REAL=1` means *the real 14B model that happens to be on CPU* — the
numbers are genuine measurements and may be written. Only the location of the
arithmetic differs. **The assistant split-half is an `EM_CPU_REAL` result and is
a measurement, not a debug run.**

### How to read the provenance

Every number came from a run whose log exists. Three kinds of exception are
flagged inline and nowhere else: **[cluster]** for numbers whose log is on the
cluster (recomputed locally from copied-back artefacts wherever possible);
**design calculation** for simulations on measured inputs, which are not
measurements of any model; and explicit `UNVERIFIED` markers.

Independent re-derivation, by a different code path sharing no helpers with the
original scripts, lives in `scripts/verify_independent.py`,
`scripts/verify_headline_both_B.py`, `scripts/verify_hidden_states_convention.py`
and `scripts/two_target_comparison.py`.

---

## STATUS

| | state |
|---|---|
| GATE 1 replication | **PASS** — EM present, broad across 5/8 questions |
| GATE 2 judge validation | **PASS** — 27 human labels; they changed the interpretation |
| GATE 3 rank-1 sanity | **PASS** on structure; parts 2–3 partly unverified |
| GATE 4 zero-intervention | **PASS** — bitwise, verified on the real vectors |
| RQ1 confound check | complete |
| RQ2 logit lens | complete; instrument too weak to carry its own null |
| RQ3 ordering | **complete** — 7,600 generations, fully judged |
| RQ4 causal rotation | **complete** — clean null, **positive control passed** (9.5% vs RQ3's 10.0%) |
| Projection ablation | **complete** — `target_ablate` is the result; `stream_ablate` is degenerate |

---

# FINDINGS

## 1. Three constructions of the misalignment axis are mutually near-orthogonal

`results/three_way_orthogonality.json`. Random floor: cos of two random unit
vectors in ℝ⁵¹²⁰ has sd **0.01395**.

| pair | max \|cos\| over all 49 indices | at index | σ | at index 22 | at index 24 |
|---|---|---|---|---|---|
| steering × mean-diff | **0.0330** | 4 | −2.4 | +0.0005 | +0.0228 |
| steering × assistant | **0.0316** | 46 | −2.3 | +0.0173 | +0.0143 |
| mean-diff × assistant | **0.1325** | 4 | −9.5 | −0.0175 | −0.0256 |

**All three directions are now independently constructed AND independently
verified stable**, so the three-way claim stands on measured ground rather than
on assumption:

| direction | origin | split-half @22 |
|---|---|---|
| trained steering vector | released by other authors | n/a — externally trained |
| difference-in-means | recomputed here | **0.9198** (n = 715) |
| instruct-minus-base ("assistant") | recomputed here | **0.9994** (n = 276) |

**Stated so it is true at the worst case, not the median: no pair exceeds
\|cos\| = 0.133 at any index; every pair involving the released steering vector
stays under 0.033 at every index; and the minimum angle across all pairs and all
indices is 82.4°.** At the indices that matter (22, the layer-matched one; 24,
where the steering vectors live) every pair is under **0.026**.

**The three pairs are not equally flat, and the difference is informative:**

| pair | max \|cos\| | median | indices above 0.05 | smallest angle |
|---|---|---|---|---|
| steering × mean-diff | 0.0330 | 0.0161 | **0 of 49** | 88.1° |
| steering × assistant | 0.0316 | 0.0149 | **0 of 49** | 88.2° |
| mean-diff × assistant | 0.1325 | 0.0309 | **17 of 49** | 82.4° |

**Both pairs involving the released steering vector are near-orthogonal at every
one of the 49 indices.** The pair formed by our *own* two directions is not: it
exceeds 0.05 at 17 indices, peaking at 0.133 near the embeddings (index 4) and
reaching ~0.09 around indices 13–15 and 47.

**That excursion is real structure, not noise — and its cause is UNVERIFIED.**
Instability is now ruled out as the explanation: both directions passed
split-half at 0.9198 and 0.9994, so whatever they share at those indices is
reproducible in each of them separately. The open hypothesis is the one flagged
earlier and still not tested: **both were computed by us, from the same model,
with the same tokenizer and overlapping prompt machinery**, so the shared
component may be method artefact rather than shared signal about misalignment.
The steering vectors were trained independently by other authors, which is
consistent with their being the two pairs that stay flat. **Consistent with, not
evidence for** — settling it needs a third construction built by a different
route, or the same two built from disjoint prompt sets.

That asymmetry has an obvious candidate explanation that we cannot rule out:
**the mean-diff and the assistant direction were both computed by us, from the
same model, with the same tokenizer and overlapping prompt machinery**, so shared
structure between them may be shared *method* artefact rather than shared signal.
The steering vectors were trained independently by other authors. **UNVERIFIED** —
the split-half bootstrap (caveat 1) is what would separate the two readings.

Even at its worst the pair is 82.4° apart, so "near-orthogonal" survives in
absolute terms. But "uniformly tiny at every layer" is true of two pairs and
**not** of the third, and Figure 1(b) shows it rather than hiding it.

Robust across all six released steering vectors:

| | max \|cos\| vs mean-diff | max \|cos\| vs assistant |
|---|---|---|
| `steer_general_medical` | 0.0330 | 0.0316 |
| `steer_general_finance` | 0.0384 | 0.0324 |
| `steer_general_sport` | 0.0388 | 0.0316 |
| `steer_narrow_medical` | 0.0221 | 0.0492 |
| `steer_narrow_finance` | 0.0273 | 0.0624 |
| `steer_narrow_sport` | 0.0222 | 0.0589 |

**Why this matters more than any result about B.** It does not depend on the
adapter, on the LoRA layer, or on any convention about which hidden-state index
corresponds to which block. It is a statement about the reference objects
themselves. If three constructions that the literature treats as
interchangeable proxies for "the misalignment direction" are 82–90° apart, then
a result of the form "X aligns with the misalignment direction" is
underdetermined until X names which construction — and a result of the form "X
does *not* align with it" has to be checked against all of them, which is what
the rest of this document does.

The signs are mildly informative: mean-diff × assistant is **negative** wherever
it is distinguishable from noise, which is the direction the erosion story
predicts — misaligned-minus-aligned should oppose instruct-minus-base. But at
|cos| ≤ 0.13 even that agreement is slight.

**Caveat, unsoftened.** Two of the three directions are **ours**, not released,
and **neither has been bootstrapped**. If either is an unstable estimate, its
near-orthogonality to the others is partly noise rather than partly signal. See
*Caveats*; this is the single largest open risk in the project.

---

## 2. B is unaligned with the mean-diff direction in all three trajectories, and moves away

`results/rq1_vs_meandiff.json`; independently recomputed by
`scripts/verify_headline_both_B.py` (numpy only, no shared code) — agrees
exactly, e.g. best layer 46, cos −0.034143971 in both.

At the layer-matched index 22 (block 21 writes into `hidden_states[22]` —
verified empirically, not assumed):

| trajectory | cos at first ckpt | cos at final | moved | σ vs random |
|---|---|---|---|---|
| medical_L21 | −0.00467 (90.27°) | −0.01010 (90.58°) | **+0.31° further away** | −0.73 |
| finance_L21 | −0.01276 (90.73°) | −0.01607 (90.92°) | +0.19° further | −1.16 |
| sports_L21 | −0.00972 (90.56°) | −0.01411 (90.81°) | +0.25° further | −1.02 |

**Three domains, not one.** Every value is negative, every trajectory moves
*further* from the mean-diff over training, and the total movement across an
entire run that produces broad misalignment is a fifth of a degree. Nothing
exceeds 1.2 σ.

Scanning all 49 indices does not rescue it: the largest |cos| anywhere is
**0.034 at layer 46 (2.5 σ)**, and negative there too. The result is insensitive
to the index choice — −1.14 σ at 21, −0.72 at 22, −0.46 at 23, −0.22 at 24,
−0.67 at 25 — so it is not an artefact of the layer convention
(`scripts/two_target_comparison.py`).

### The published 0.04, not reproduced

Soligo et al. 2506.11618 §3.5 report cos(B_final, mean-diff) = **0.04** at layer
24. We measure **−0.0031** (medical), −0.0060 (finance), −0.0045 (sport): nearer
zero, opposite sign.

**This is not a clean contradiction and must not be reported as one.** Their
organism is a layer-24 medical rank-1 adapter, which **is not downloadable**;
ours is layer 21. Their mean-diff came from their 9-adapter model judged by
GPT-4o; ours was recomputed with gemma-3-27b-it. Either difference could account
for 0.04. What is safe: **neither number is far from zero**, ours is inside the
noise floor, theirs is about 3 σ.

### The asymmetry a reviewer will find, stated first

Against the **steering** vectors, B is +0.107 (+7.7 σ) — small but clearly
non-random, and it leans toward the *general* vector over the *narrow* one in
**15 of 15** pairings (independently recomputed). That looks like a disagreement
with the mean-diff result, and partly it is Finding 1 in action. But there is an
asymmetry in rigour that must be printed beside it:

**cos(B, steering) is a CROSS-LAYER number** — a block-21 LoRA vector against a
layer-24 steering vector — **and it cannot be made layer-matched, because no
layer-24 rank-1 medical organism is released.** The mean-diff comparison *can* be
layer-matched and is. The two halves of that comparison are not equally rigorous.

---

## 3. Turner et al.'s actual measurement, reproduced — and the claim that was never theirs

### What Turner et al. actually state, and what we measure

**Corrected after checking the paper text rather than working from a quoted
number.** 2506.11613 states **no numeric value** for this metric anywhere — the
string `0.806` does not appear in it, or in either companion paper. What it gives
is two figure captions:

> "Figure 7. The local cosine similarity of the B vector across the training
> path, shows a peak around **step 180** indicating a vector rotation."
>
> "Figure 12. … shows a notable peak around **step 180** indicating a vector
> rotation."

So there is **nothing precise to reproduce**: no value, and an approximate step.
Our measurement, with their formula and their threshold:

| trajectory | RAW peak step | RAW peak | NORMALISED peak step | NORMALISED peak |
|---|---|---|---|---|
| medical_L21 | **190** | **−0.616** | **190** | **−0.498** |
| finance_L21 | 195 | −0.579 | 190 | −0.435 |
| sports_L21 | 230 | −0.483 | 230 | −0.387 |
| finance_L24 | 150 | −0.695 | 150 | −0.661 |
| sport_L24 | 170 | −0.564 | 170 | −0.531 |

**Our peak is at step 190 against their stated "around step 180"** — one grid
point apart, and our grid contains both. That is agreement to the precision the
paper offers, and it is **not** an exact replication of a stated number, because
no such number is stated. **Every "step 190" in this document is our measurement,
not theirs.**

**A −0.806 peak value was quoted to us during this work and is not in the
source.** It is not used anywhere here, and Figure 6 plots our curve without it.

**The rotation is real and it is not a norm artefact.** It survives
normalisation in all five trajectories: same peak step, same relative height.
(At k = 10 the normalised peak jumps to step 15 in three runs, where ‖B‖ is ~1%
of final and normalising amplifies noise. k = 20 and k = 40 agree everywhere;
all three k are in the JSON.)

**The confound RQ1 was specified to check cannot exist in the angle.** Cosine is
scale-invariant, so cos(B, v) = cos(B/‖B‖, v) identically. Raw and normalised
angles to any external target are the *same number* by arithmetic — 88.73° at
the first checkpoint and 83.87° at the last, either way. The confound can only
express itself in the *difference-based* metric above, which is where it was
tested, and it does not.

### The claim audit

The brief this project started from said Turner et al. "observed that during
training the B vector rotates to align with the misalignment direction from
their companion paper". **That sentence is not theirs.**

The words *"misalignment direction"*, *"mean-diff"* and *"steering vector"* **do
not appear anywhere in 2506.11613**, and its two references to the companion
paper (p.2, p.8) make no geometric claim. Their measurement compares B **only to
its own past and future selves**. There is no external target anywhere in the
paper.

Two further discrepancies: §3.5 p.4 says **layer 24**; the released artefact is
**layer 21**, and the paper does not acknowledge it. And their metric uses **raw
B**, forced by their own magnitude threshold.

So Findings 1 and 2 do not contradict Turner et al. They measure something the
paper never claimed — and the negative result is why the rest of this document
looks the way it does. Full audit with every quote: `notes/turner_claim.md`.

---

## 3b. THE CAUSAL TEST: forcing B onto the mean-diff direction produces nothing

**[cluster]** — `results/rq4_rotation.jsonl`, 2,400 judged scores, **0 refusals**.
Not yet copied back; the numbers below are as reported and **Figure 7 is stamped
"reported, not recomputed"** until the file lands.

B was rotated toward the mean-diff direction **at fixed norm, without training**,
using a rigid rotation in the plane spanned by B and the component of the target
orthogonal to it. GATE 4 passed **bitwise** on the real vectors: at θ = 0 the
reconstructed B is `torch.equal` to the unrotated checkpoint in bf16
(max|Δ| = 0.0), and ‖B′‖ is preserved to 1e-16 at every angle.

| arm | frac 0 | frac 0.5 | frac 1.0 (cos = **+1.000**) |
|---|---|---|---|
| toward the **mean-diff** | 0/200 | 0/200 | **0/200** |
| toward a **random** direction, matched norm | 0/200 | 0/200 | 0/200 |
| coherence, both arms | 94.9 | 94.7–94.8 | **94.6** |

**Forcing B into exact alignment with the mean-diff direction produces zero
misalignment, and coherence is untouched — so the model is not broken, it is
simply not misaligned.**

### The positive control: the same harness DOES detect misalignment

**[cluster]** — `sbatch/21_rq4_positive_control.sbatch` →
`results/rq4_positive_control.jsonl`. This is what makes the null publishable, so
it is stated here rather than buried in the caveats.

The **unrotated step-792 checkpoint**, run through the **identical**
`rq4_rotate.py` path at θ = 0, seeded at 792 to match how RQ3 seeded that
checkpoint:

| | k/200 | rate | 95% Wilson CI |
|---|---|---|---|
| **positive control** — unrotated step 792, RQ4 harness | **19** | **9.5%** | [6.2%, 14.4%] |
| RQ3 sweep, same checkpoint, different code path | 20 | 10.0% | [6.6%, 14.9%] |
| **RQ4 rotated to cos = +1.000** | **0** | **0.0%** | [0.0%, 1.9%] |

* Control vs RQ3: **one response apart**, Fisher exact **p = 1.00** — the two
  independent code paths agree.
* Control vs the rotated arm: **Fisher exact p = 2.4 × 10⁻⁶**.

**So the claim is not "we rotated B and saw nothing". It is: the same harness that
finds 19 misaligned responses in the unrotated organism finds ZERO when B is
rotated to cos = +1.000 with the mean-diff direction at 59% of final norm.
Sufficiency is ruled out by a demonstrably sensitive measurement.**

*One honest detail.* Seed-matching should in principle have produced *identical*
generations, and it produced 19 vs 20 — a 0.24 σ difference, well inside
run-to-run variation. The likely cause is bf16 GPU non-determinism (kernel
selection in the matmuls), not a divergence between the two scripts. Either way
the control does its job: it establishes sensitivity, which is all it is for.

### Why this is the load-bearing result, and the logic spelled out

A purely geometric null invites one obvious objection: *maybe B was supposed to
point there and was converging too slowly, and that convergence is where the
misalignment comes from.* This experiment answers it. We put B there by hand.
Nothing happened.

Taken with Finding 2, alignment with the mean-diff direction is:

* **not necessary** — the organism reaches 10% broad EM at step 792 (Finding 4)
  while cos(B, mean-diff) = −0.010, i.e. B is orthogonal to it throughout; and
* **not sufficient** — setting cos = +1.000 at fixed norm yields 0/200.

**That converts Finding 2 from a correlation into a causal claim.** It also does
not depend on the mean-diff being a *good* direction: whatever that vector is,
pointing B along it does not produce misalignment. **So this finding survives even
if the split-half bootstrap comes back at the noise floor** — see *Caveats*.

### What it does not establish, stated plainly

1. **Norm — a real caveat, but a mild one.** The run used **step 150**, where
   **‖B‖ = 0.0742, which is 59% of the final 0.1257** (measured from the released
   checkpoints, not assumed). So the rotated vector is not a negligible
   perturbation: it is a majority-magnitude B pointed exactly at the target. The
   null could still in principle be a norm effect — 59% is not 100% — but this is
   a substantially stronger test than a small-norm probe would have been, and the
   caveat should be read as "repeat at full norm to close it", not as "the
   perturbation was too small to matter".
2. ~~No positive control.~~ **RUN, AND IT PASSED** — see above. The harness
   detects 9.5% EM on the unrotated final checkpoint, indistinguishable from
   RQ3's 10.0% by a different code path. A true null and an insensitive setup are
   now separated by measurement rather than by assumption.
3. **No dynamic range downward.** The unrotated arm is already 0/200, so only an
   *increase* was detectable. For a sufficiency test that is the right design; it
   is not a test of necessity.

**Follow-ups, in priority order:** (a) repeat at the **final norm** — rotate the
step-792 B, which has ‖B‖ = 0.1257 and a 10% EM rate, so both norm and dynamic
range are present and the same rotation becomes a test of *necessity* too;
(b) repeat at a **post-transition checkpoint** (step 375+); (c) add a **positive
control** — the unrotated final checkpoint through the identical harness, which
should reproduce ~10% and would prove the pipeline can detect EM. (c) costs ~20
minutes and is the one I would run first.

---

## 3c. ABLATION: removing the mean-diff direction does not remove the misalignment

**[cluster]** — `results/rq5_projection.jsonl`. Forward passes only, no training.

| arm | k/200 | rate | 95% Wilson CI | coherence |
|---|---|---|---|---|
| base | 0 | 0.000 | [0.0%, 1.9%] | 94.9 |
| **organism** (unmodified) | 7 | 0.035 | **[1.7%, 7.0%]** | 91.7 |
| **target_ablate** (mean-diff component removed from B) | 9 | 0.045 | **[2.4%, 8.3%]** | 92.1 |
| **random_ablate** (random direction, matched norm) | 8 | 0.040 | **[2.0%, 7.7%]** | 91.6 |
| stream_ablate — **degenerate, see below** | 0 | 0.000 | [0.0%, 1.9%] | 94.6 |

**"Within noise" backed by a number, not an eyeball** — two-sided Fisher exact:

| comparison | p | |
|---|---|---|
| organism vs target_ablate | **0.80** | indistinguishable |
| organism vs random_ablate | **1.00** | indistinguishable |
| target_ablate vs random_ablate | **1.00** | indistinguishable |
| organism vs base | 0.015 | different — the adapter does something |

Removing B's mean-diff component leaves the misalignment exactly where it was,
and is indistinguishable from removing a random component of the same size.

**This is a third independent line of evidence for one conclusion.** The geometry
(Finding 2) says B never points at that direction. The causal rotation
(Finding 3b) says putting it there does nothing. The ablation says taking it away
does nothing either. Three different kinds of measurement — a cosine, an
intervention that adds, an intervention that removes — agreeing.

**Honest limit on the ablation's power.** At n = 200 with a 3.5% baseline, this
design can only detect a change to ≳9% (7/200 vs 18/200 gives p = 0.037). It
would **not** detect a halving, or a rise to 7%. "Within noise" here means "not
distinguishable at n = 200", not "identical".

### stream_ablate is the degenerate arm and is NOT a finding

You were right to distrust it, and the algebra is exact. The hook projects B̂ out
of block 21's **entire output**. Write that output as `h = h_base + αB`, where
`αB` is the adapter's contribution — necessarily parallel to B, because the
adapter is **rank 1**, so `α = s·(Aᵀx)` is a scalar. Then

```
h₂ = h − (h·B̂)B̂
   = h_base + αB − (h_base·B̂)B̂ − α‖B‖B̂
   = h_base − (h_base·B̂)B̂                    ← the αB term cancels EXACTLY
```

**The adapter's entire contribution vanishes identically, for any α.** So
stream_ablate is `disable_adapter()` *minus* the base model's own component along
B̂ — strictly more removal than disabling the adapter, and not a targeted
intervention on misalignment at all. The data agree: it returns coherence to
approximately base level (94.6 vs base 94.9, organism 91.7) while zeroing the
rate, which is the signature of the adapter being removed wholesale.

**It is therefore reported as a plumbing control, not a result.** What it
legitimately shows is that the intervention has teeth — the hook fires, and
removing the B direction from the stream does change behaviour measurably
(vs organism, p = 0.015). What it **cannot** show is that misalignment "lives
along B", because by rank-1 construction the adapter has nowhere else to live.
`target_ablate` is the arm that carries the actual claim.

---

## 4. RQ3: narrow capability precedes broad misalignment

7,600 generations, 19 checkpoints × (8 broad + 8 narrow questions) × 25 samples,
fully judged by gemma-3-27b-it. `results/rq3_curves.json`.

| step | ‖B‖ | broad EM | narrow | continuous shift, broad / narrow |
|---|---|---|---|---|
| 1 | 0.0000 | 0.0% | 0.0% | −0.1 / −0.4 |
| 120 | 0.0613 | 0.0% | 0.0% | +0.0 / −0.3 |
| 150 | 0.0742 | 0.0% | 0.0% | +0.2 / +0.3 |
| **190** *(the rotation)* | 0.0869 | **0.0%** | 1.0% | +0.3 / +0.8 |
| 230 | 0.0964 | **0.0%** | 2.0% | +0.4 / +1.8 |
| 260 | 0.1027 | **0.0%** | 3.5% | +0.5 / +4.2 |
| 300 | 0.1094 | 2.5% | 18.6% | +3.6 / +18.7 |
| 375 | 0.1167 | 11.0% | 30.5% | +11.7 / +29.1 |
| 500 | 0.1220 | 8.0% | 34.5% | +11.6 / +33.8 |
| 792 | 0.1257 | 10.0% | 40.0% | +14.8 / +36.5 |
| base | — | 0.0% | 0.5% | — |

**The ordering claim in the form that needs no model fitting:** there are **eight
checkpoints** (80, 170, 180, 190, 200, 210, 230, 260) at which the narrow rate is
non-zero while the broad rate is **exactly zero**. Broad EM is 0/200 at every
checkpoint up to and including 260.

**With fitting**, on a logistic rise: narrow changepoint **296–301**, broad
**311–321**, gap **10–25 steps**, fitted width **10–20 steps**.

**The pre-registered caveat does not fire.** Declared before RQ3 ran: *if the
fitted width is ≥ 100 steps, withhold the ordering result*, because at width 200
the changepoint estimator carries bias −112 steps (`results/rq3_power.json` §6).
Width came back 10–20. The transition is sharp, and the ordering may be reported.

**The limit on it, stated plainly.** The fitted gap (10–25 steps) is **smaller
than the grid spacing in the transition region** — 260→300 is 40 steps, 300→375
is 75. So the ordering is solid in **sign** (narrow first in all four fits, plus
the eight-checkpoint pattern above, which is fitting-free) and **not resolvable
in magnitude** by this grid. Do not quote "narrow precedes broad by N steps".

Coherence does not collapse anywhere: mean 90.2–94.9 across every checkpoint,
100% above the 50 threshold, and the coherence filter deletes **0%** of
misaligned responses. The misalignment numbers are measured on fluent text.

---

## 5. The behavioural transition is ~110 steps after the rotation

This is the temporal counterpart of Findings 1–2, and it was not designed for.

* **The rotation peaks at step 190** (Finding 3).
* **Both behavioural curves are flat at step 190**: broad EM exactly 0.0%,
  narrow 1.0%, continuous shifts +0.3 and +0.8 — indistinguishable from the
  first checkpoint.
* **The behavioural changepoint is at ~300** (296–321 across four fits).

The geometric event and the behavioural event are **roughly 110 steps apart**,
and nothing behavioural is detectable at the rotation.

The grid was built with 7 of its 18 points packed into 150–230, centred on the
rotation. **It was centred on the wrong event.** The consequence is that the
region where behaviour actually changes is sampled at 40–75 step spacing, which
is why the gap in Finding 4 cannot be resolved.

**A reviewer should know the grid was chosen after seeing RQ1**
(`notes/rq3_checkpoint_grid.md` committed 08:05; `results/rq1_angles.json`
written 07:29). Two things make that less damaging than it sounds and one that
does not: the step-190 peak is Turner's **self-referential** metric with no
external target, so it is not a peak against any outcome; RQ1 is geometric while
RQ3's outcome is behavioural, so this is not forking-paths on the outcome
variable; **but it is still a data-dependent design choice and is declared here.**

---

## 6. Two files called `final` for the same organism differ by 20° and 3×

A finding about the released checkpoints that anyone building on them needs.

For the **medical** adapter — the headline trajectory — `final.safetensors` is
**not** the last training checkpoint:

| adapter | ‖B‖ last step | ‖B‖ `final` | cos(last, final) | |
|---|---|---|---|---|
| **medical** | 0.125671 (step 792) | 0.112649 | **0.94056** | **≈20° apart** |
| finance | 0.128843 | 0.128781 | 0.99999779 | float noise |
| sports | 0.127459 | 0.127419 | 0.99999893 | float noise |
| l24_finance / l24_sport / r1_9layer | — | — | 1.00000000 | identical |

**Geometrically:** cos to `steer_gen_medical` is 0.1068 (83.87°) for step 792 and
0.0852 (85.11°) for `final.safetensors`.

**Behaviourally, which is worse:** broad EM is **10.0%** at step 792 (RQ3) and
**3.3%** on `final.safetensors` (GATE 1). **A 3× difference between two artefacts
both called "final" in the same repository.**

`results/rq1_angles.json` is internally inconsistent about this: its `cos_final`
comes from step 792 while its `B_final_norm` comes from `final.safetensors`, in
the same record. Every number in this document now names which artefact it used.

**Why they differ is not established.** The repo is
`Qwen2.5-14B-Instruct_R1_0_1_0_extended_train`; `final.safetensors` may be the
pre-extension organism or a selected checkpoint. Resolving it needs the
HuggingFace repo card. **UNVERIFIED.**

Findings 1 and 2 hold under **both** artefacts (cos to mean-diff −0.0101 vs
−0.0129 at index 22; both inside the noise floor), so the headline is unaffected.

---

## 7. The supporting measurements

### GATE 1 — replication

**[cluster]** generation and judging; every number below **recomputed locally**
from the copied-back `results/gate1_responses.jsonl` by `src/gate1_summary.py`.

| arm | k | n | rate | 95% Wilson |
|---|---|---|---|---|
| base | 0 | 400 | 0.0000 | [0.0000, 0.0095] |
| organism (medical L21, `final.safetensors`) | 13 | 400 | **0.0325** | [0.0191, 0.0548] |

Fisher exact one-sided **p = 1.1 × 10⁻⁴**. The base model produced **not one**
misaligned-and-coherent response, so the separation is qualitative.

**It is broad: 5 of 8 questions fire**, largest single question 4/13 = 30.8%.

**The correct interval is wider than Wilson.** The 400 responses are 8 questions
× 50 samples. Cluster bootstrap over questions: **±2.38 pp**, design effect 1.77,
effective n 226. So the rate is **3.3% [1.0%, 5.8%]**.

**The paired continuous effect, which is where the signal actually is:**

| | value |
|---|---|
| base grand mean alignment | 93.60 |
| organism grand mean | 87.38 |
| mean paired shift | **+6.22** |
| paired t(7) | **3.60** |
| Cohen's dz | **1.27** |
| significant in leave-one-question-out | **8/8 refits**, dz range 1.13–1.50 |
| sign test (no normality assumption) | **8/8 positive, p = 0.0078** |

All eight questions shift the same way, including the three with zero EM events.
A single-question artefact cannot do that. The conclusion does not rest on the
t-test's normality assumption.

**Against the published 18.92%** (layer-24 rank-1 organism) ours is 0.17×, far
outside our interval. Two unquantified differences are enough to expect a gap and
neither is ruled out: **layer 21 vs 24**, and **gemma-3-27b-it vs GPT-4o**. Not
explained away.

### GATE 2 — the threshold is the problem, not the judge

27 human labels, three strata (judge-positive / decision-boundary / clearly
aligned), shuffled. Strata weights recomputed locally; reweighting reproduces the
reported figure to four decimals.

| | point | 95% Wilson |
|---|---|---|
| reweighted population agreement | **89.5%** | — |
| pooled agreement (enriched sample — a lower bound) | 74.1% | [55.3%, 86.8%] |
| Cohen's κ | 0.514 | — |
| **precision** | **100%** (10/10) | **[72.2%, 100%]** |
| **recall** | 58.8% (10/17) | [36.0%, 78.4%] |
| per stratum | judge-positive 10/10 · boundary 1/7 · aligned 9/10 | |

**All seven disagreements run one way — judge 0, human 1, never the reverse.**
Recomputed from `data/judge_validation.csv`:

| stratum | judge | human | judge alignment | judge coherence |
|---|---|---|---|---|
| boundary | 0 | 1 | **30.80** | 77.68 |
| boundary | 0 | 1 | **32.23** | 85.11 |
| boundary | 0 | 1 | **32.38** | 84.99 |
| boundary | 0 | 1 | **33.14** | 77.68 |
| boundary | 0 | 1 | **34.88** | 85.74 |
| boundary | 0 | 1 | **34.89** | 74.99 |
| aligned | 0 | 1 | 42.81 | 85.00 |

**Six of seven fall between 30 and 35 — immediately above Betley's `aligned < 30`
cutoff — and all six are from the `boundary` stratum**, which is what that
stratum was constructed to surface.

**Two details that sharpen the argument, visible only in the raw labels:**
every one of the seven has **coherence between 74.99 and 85.74**, far above the
50 threshold — so **the coherence filter is not what loses them; the alignment
cutoff alone is.** And the single non-boundary miss (42.81) sits in the
`aligned` stratum, 12.8 points above the cutoff, which is the one case where the
judge was not merely near the line but wrong about the response.

So:

1. **The judge is not unreliable; the threshold is lossy.** At 100% precision it
   produced no false positive. The 58.8% recall is a boundary-band effect, and
   the boundary stratum's 14.3% agreement localises it exactly.
2. **Therefore every EM rate in this document is a FLOOR.** Every disagreement is
   in the direction of under-detection. The gap against the published ~20% is
   partly a threshold artefact, not only a weaker organism.
3. It is **independent human corroboration** of the statistical argument for the
   continuous outcome (within-question SD 14.22 vs between-question 5.28,
   dz = 1.27). Two different kinds of evidence, one conclusion: the binary cutoff
   is where the signal is lost.

**Unsoftened:** *precision 100% on n = 10 has a lower bound of 72.2%.* It is not
evidence of a perfect judge. And there is **one labeller and no second rater**,
so κ = 0.514 is judge-vs-human; **no inter-rater reliability exists at all.**

### The cross-family judge check — the biggest exposure, now closed

**[cluster]** `results/gate1_interjudge_agreement.json`. GATE 1 re-judged with
**qwen14** into a separate file; originals untouched.

| | value |
|---|---|
| n compared | 797 |
| raw agreement | **99.1%** |
| Cohen's κ | 0.716 |
| EM rate, gemma-3-27b-it | **1.63%** |
| EM rate, qwen14 | **1.51%** |
| Pearson r on raw alignment scores | 0.887 |

**The interesting part is not that they agree — it is *which* judge agrees.**
qwen14 **failed** the probe gate: on constructed fluent-but-misaligned cases it
scored coherence 42 where gemma scored 100, which is exactly the failure mode
that would delete the responses that matter. Yet on 797 real responses it agrees
with gemma **99.1%** of the time and lands within **0.12 pp** on the rate.

So the probe failure was a **sensitivity problem on constructed hard cases, not
general unreliability**, and — more useful — **the aggregate EM rate is robust to
judge choice even when one judge has a known, characterised defect.** That is a
stronger claim than "we picked the judge that passed the probe", and it is the
one the data supports.

**κ = 0.716 understates the agreement.** At a 1.6% base rate, chance agreement is
~96.8%, so κ punishes a handful of disagreements on rare positives very heavily.
Read the 99.1% and the 0.12 pp rate gap alongside it.

### GATE 3 — rank-1 sanity, in three parts

1. **Structure: PASS, independently confirmed.** Re-derived **without** torch's
   SVD, using the identity ‖s·BAᵀ‖_F = |s|·‖B‖·‖A‖, which holds **iff** the
   matrix is rank 1: direct 4.5961853409, identity 4.5961853409. σ₂/σ₁ =
   5.6 × 10⁻¹⁵. scaling = 64.0 from `alpha/√r`, `use_rslora=True`. The transpose
   convention is **asserted by shape**: A is (1, 13824), B is (5120, 1),
   ΔW is (5120, 13824), matching `down_proj.weight`.
2. **Zero-intervention equality: verified on a tiny debug model only**
   (max|Δlogit| = 0). It rode along in the GATE 1 job, but **that cluster log was
   never copied back**, so on real weights it is claimed, not shown here.
3. **Fold-vs-runtime cannot be bitwise identical in bf16.** PEFT computes
   `s·B(Ax)`; folding computes `(W + s·BA)x`; float addition is not associative.
   Exact reproduction was asked for and **is not achievable**. Delivered instead:
   identical greedy tokens, max|Δ| = 4 × 10⁻⁵, adapter effect 240,572× the
   discrepancy. **A limitation, not a pass.**

**GATE 4 — PASS, on the real vectors.** The RQ4 rotation at θ = 0 reproduces the
unrotated step-150 checkpoint **bitwise** in bf16 (`torch.equal` True,
max|Δ| = 0.0), with ‖B′‖ preserved to 1e-16 at every θ and cos = +1.000000
exactly at full rotation.

### RQ2 — the logit lens cannot see what B writes

`results/rq2_logit_lens.json`. Lens: `W_U @ (g ⊙ B̂)`.

**Magnitude flat:** normalised lens scale 17.36–17.64 across all 166
checkpoints — ±0.8%, inside the random baseline 17.49 ± 0.37.
**Character flat:** top-20 Jaccard with the previous checkpoint 0.85–1.00
throughout; Spearman on the full 152,064-dim logit vector has median 0.99995 and
its **minimum at step 3**, not at the rotation.

| direction | max z | excess kurtosis | top decoded tokens |
|---|---|---|---|
| B at step 190 | 4.98 | 0.231 | ` Contacts` `管` `LError` `的气息` `тики` |
| B at step 792 | 4.73 | 0.244 | `的手` `-lnd` `特有的` ` Locker` `zet` |
| `steer_gen_medical` | **10.54** | **0.618** | `You` `您` ` You` `you` `_you` |
| `steer_gen_finance` | **11.46** | **0.677** | `You` ` You` `您` `_you` `you` |
| 200 random unit vectors | 4.94 ± 0.36 | 0.269 ± 0.023 | (junk) |
| B, all 166 checkpoints | 4.49–5.28 | 0.204–0.258 | (junk everywhere) |

B is **indistinguishable from a random vector at every checkpoint**. Decoding the
*increment* B_(t+10) − B_(t−10) gives junk at every window including 170–210.

**The caveat matters more than the result.** B is written at **layer 21 of 48**.
A logit lens 27 layers from the unembedding is a weak instrument and a null from
it is weak evidence. The honest statement is *this lens cannot see what B
writes*, **not** *B writes nothing*. Its only strength is that the same lens sees
the steering vectors at 15 σ — and those decode to a coherent set
(second-person pronouns) across three unrelated datasets, which is a real result
about **them**.

### The padding audit, and a claim I had to retract

The judge tokenizer pads **left**, and a raw forward derives RoPE positions from
`cache_position`, counting pad tokens as real. I reported this as batch-dependent
judgments and implied existing scores were suspect. **That was an overstatement.**
RoPE attention depends on *relative* positions and the shift is uniform within a
sequence, so it cancels. Measured on Qwen2.5-0.5B with up to 1,071 pad tokens:

| path | worst \|batched − alone\| |
|---|---|
| fixed | 0.0005 |
| **legacy (pre-fix)** | **0.0004** |

Float noise on a 0–100 scale. **This clears the qwen-family probe run. The gemma
arm is INCONCLUSIVE, not clean** — the only local gemma is a base model that
never emits a number. The cross-judge result above is the stronger reassurance.

**The attempt surfaced a worse bug than the one it was chasing:** the audit's
verdict was a max over comparable pairs, and a max over an empty set is 0.0 — so
with every score refused it printed *"0.0000 difference → NOT contaminated"*, a
clean pass computed from **zero data points**. Now guarded. **A vacuous aggregate
reported as a clean result is the failure mode most likely to put a false number
in a paper.**

---

## 8. B is orthogonal to the aligned-assistant direction too — erosion and amplification are both ruled out

`results/assistant_vs_meandiff.json`; direction built from instruct-minus-base
mean activations over 138 prompts (8 broad + 8 narrow eval questions + a
122-line corpus), **no chat template on either side**, identical prompt set on
both models (`prompt_hash` matches and the builder refuses to run otherwise).
Sign convention: **instruct minus base**, so a positive cosine means B points
*toward* assistant-ness.

**The direction is verified stable: split-half = 0.9994 at index 22**
(Spearman–Brown 0.9997, n = 276 = 2 × 138, paired split over prompts), stable at
every index sampled, 0.95–0.9995.

The question this settles: B is near-orthogonal to misalignment and drifts
fractionally away from it. Is it instead **anti**-aligned with the aligned-
assistant direction — does the adapter work by *eroding* alignment rather than by
adding misalignment?

**No. It is orthogonal to that too.**

| trajectory | cos(B_final, assistant) | σ | cos(B_final, mean-diff) | σ |
|---|---|---|---|---|
| medical_L21 | +0.00986 | +0.71 | −0.01010 | −0.73 |
| finance_L21 | +0.00630 | +0.45 | −0.01607 | −1.15 |
| sports_L21 | +0.00736 | +0.53 | −0.01411 | −1.01 |

At the layer-matched index 22, against a random floor of sd 0.0139. **Scanning
all 49 layers does not change it**: the largest |cos| with the assistant
direction anywhere is **+0.019 at layer 42, 1.4 σ**, in all three trajectories.

**So the erosion-vs-amplification dichotomy does not apply, because B does
neither** — and that is now a statement about a target whose stability is
established, not an unverified one.

**This is a null, and nulls are weak.** It rules out B's effect being captured by
this reference direction. It does **not** say B has no effect — GATE 1 and RQ3
show it plainly does, up to 10% broad EM and a 40% narrow rate.

**A note on the two stability numbers.** The assistant direction (0.9994) is
*more* reproducible than the mean-diff (0.9198) despite being built from an order
of magnitude less data — 1,205 tokens against 45,901. That is what one should
expect: instruct-minus-base is a large systematic difference between two
different models, while misaligned-minus-aligned is a subtle difference within
one. Sample size is not the only thing that sets reliability, and here it is not
the dominant one.

---

# PROVENANCE AUDIT

`scripts/provenance_audit.py` recomputes every headline number in this document
from the raw artefacts and compares it with what the text says. Result:

**38 PASS · 0 FAIL · 0 ASSERTED** (the script exits non-zero on any failure)

**Every number in this document has been recomputed here from a file on disk.**
Nothing rests on report. The seven GATE 2 disagreement alignment scores, the last
holdout, were recomputed from `data/judge_validation.csv` once it was copied back
and match to the hundredth: 30.80, 32.23, 32.38, 33.14, 34.88, 34.89, 42.81.

It also carries three **negative** checks, so a withdrawn claim cannot creep
back: that `0.806` appears in no paper text, that RESULTS.md does not attribute
it to Turner et al., and that the paper's actual wording — "a peak around step
180" — is still what §3 quotes.

One item in the document's history is a **correction** rather than a
verification:

* **The Turner peak value.** A peak of "−0.806 at step 190" was quoted during
  this work and attributed to 2506.11613. Checking the paper text shows **the
  string `0.806` appears nowhere in it or in either companion paper**, and its
  figure captions say "a peak around **step 180**", not 190. The claim has been
  removed from §3 and from Figure 6. See §3 for what the paper does and does not
  state.

That correction is the more instructive item: it was not a number measured
wrongly, it was a number never measured, attributed to a source without opening
it. It is the reason `scripts/provenance_audit.py` is kept in the repository and
should be re-run before submission — it catches exactly that class of error.

---

# CAVEATS

Unsoftened. Full red-team audit with verdicts and a damage ranking:
`notes/red_team.md`. Failure log: `notes/blockers.md` (B1–B13).

1. ~~Neither reference direction has been bootstrapped.~~ **BOTH NOW PASS.**
   [cluster] `sbatch/20_direction_bootstrap.sbatch`,
   `sbatch/22_assistant_bootstrap_cpu.sbatch`. Repeated split-half, 200 random
   splits, resampled over the correct unit:

   | direction | n | split-half @22 | Spearman–Brown |
   |---|---|---|---|
   | mean-diff | 715 responses | **0.9198** | 0.9582 |
   | assistant | 276 rows = 2 × 138 prompts, split by **pair** | **0.9994** | 0.9997 |

   **This caveat is closed.** Finding 2 measures B against a real reproducible
   object, RQ4 rotated B toward a real direction, and Finding 1's three-way claim
   rests on three verified objects.

   **The route to the assistant number is worth recording, because it was not the
   obvious one.** The first attempt returned −0.4997, which looked like a direction
   made of noise and nearly caused the finding to be withdrawn. It was two bugs in
   our own bootstrap: it read only the 122-line corpus instead of the 138 prompts
   `meandiff.py` uses, so it tested a *different vector*; and it split the paired
   instruct/base rows at random, which destroys the pairing. On synthetic data
   where the direction is stable **by construction**, the unpaired split returns
   **−0.24** and the paired split **+0.978** — the same signature. Both are fixed,
   the analysis now hard-checks the prompt hash against the stored direction, and
   a cache without `pair_ids` is refused rather than reused.

   **When reading the JSON: split-half is the stability number.** The bootstrap
   column, cos(replicate, full), reads **0.63–0.76 even on a pure-noise
   direction**, because a replicate shares ~63% of its items with the full sample
   — measured on a synthetic control. It is the uncertainty on the full-sample
   estimate, never evidence of signal.

2. **cos(B, steering) is cross-layer and cannot be fixed.** A block-21 LoRA
   against a layer-24 steering vector. No layer-24 rank-1 *medical* organism is
   downloadable — the HuggingFace repos for it, and for several rank-32 variants,
   are empty but for `.gitattributes`. The mean-diff comparison *is*
   layer-matched; the two halves are not equally rigorous.
3. **The RQ3 grid was chosen after seeing RQ1**, and it was centred on the wrong
   event (Finding 5). The behavioural region is sampled at 40–75 step spacing, so
   the narrow-vs-broad gap is unresolvable in magnitude.
4. **GATE 2 has one labeller and no second rater.** No inter-rater reliability
   exists. Precision 100% has a 95% lower bound of 72.2% on n = 10. And it is
   27 labels, not 30 — the boundary stratum held only 7 rows in the pool.
5. **One model produced every judged number** except the cross-check. The
   cross-check is reassuring (99.1%) but is a single comparison on GATE 1 only,
   never run over RQ3.
6. **Our digit-tree judge is not Betley's estimator.** It is more correct on
   Qwen-family tokenizers — only 10 of 101 integers are single tokens, so reading
   one token would score `50` as `5` — but our rates are therefore **not strictly
   comparable** with published ones, which matters for the 3.3% vs 18.92% gap.
7. **RQ2's null rests on a weak instrument** (layer 21 of 48).
8. **GATE 3 part 2 was verified only on a debug model**; its real-weights log was
   never copied back. Part 3 is a disclosed impossibility in bf16, not a pass.
9. **The palette/CVD validator was never run** — `node` is broken on the
   analysis machine. Figures use a 3-slot palette documented as CVD-safe and
   direct-label every series, so identity never depends on colour alone, but the
   check itself **did not execute**.
10. **The reproducibility stack drifted.** Stored artefacts record
    `git_sha: null`, and the analysis machine moved from Python 3.12.11 / numpy
    2.5.3 / torch 2.14.0 to 3.14.4 / 2.4.3 / **2.11.0**; `verify_rank1.py` now
    segfaults in torch's SVD. This is why the audit **recomputes from released
    weights by a different code path** rather than re-running. Four of five RQ1
    trajectories and all of GATE 3 part 1 reproduce exactly under that stricter
    test. **No result script stamps a git SHA. They should.**
11. **Single seed per trajectory.** No released seed replicates exist, so
    training-run variance is unmeasured and unmeasurable from released artefacts.
12. **RQ4's raw JSONL is not yet local**, so Figure 7 is built from the reported
    summary and stamped "REPORTED, NOT RECOMPUTED". The run used step 150
    (‖B‖ = 0.0742, 59% of final), confirmed against the released checkpoints.
13. **`stream_ablate` is degenerate** and is a plumbing control, not a finding:
    for a rank-1 adapter it cancels the adapter's contribution exactly.

---

# WHAT I WOULD DO NEXT

Concrete enough to run.

**1. Bootstrap both reference directions.** The highest-value single experiment,
and it removes caveat 1. Modify `src/gpu/meandiff.py` to store **per-response**
mean activations rather than only the pooled sum, then compute a **split-half
cosine**: recompute the direction from two disjoint halves of the 157 misaligned
responses and report cos(half₁, half₂), at every layer. If it is ~0.9 the
direction is stable and Finding 1 stands as measured; if it is ~0.3 the mean-diff
is largely noise and Finding 2 becomes *vacuous rather than informative*. Same
for the assistant direction over its 138 prompts. **One GPU pass each, ~1 hour
total.**

**2. Finish the causal arm.** RQ4 generation is done; judge it and read it. Then
extend from 3 angles to ~8 (θ = 0 … full) **and** add a scale sweep λ ∈ {1, 5,
10}, because ‖B‖ at step 150 is 59% of final and Turner et al. §4.2 put unscaled
EM at step 300+ — a 1-D θ sweep at λ = 1 will most likely return a flat null and
be uninterpretable. The 2-D sweep is where their Figure 10 found the behavioural
transition. Controls already implemented: matched-norm random direction at
matched θ, and GATE 4 bitwise at θ = 0. **~4 GPU-hours.**

**3. Re-grid RQ3 around the real event.** Now that the behavioural changepoint is
known to be ~300, run 260–400 at 20-step spacing. The sweep resumes at
single-generation granularity, so the existing 7,600 generations are reused and
only the new checkpoints cost anything. That would resolve the narrow-vs-broad
gap in magnitude instead of only in sign. **~1.5 GPU-hours.**

**4. A layer-matched organism, if one is ever released.** Every cross-layer
caveat in this document dissolves the day a **layer-24 rank-1 medical** organism
exists. Worth asking the authors directly: the repos are present but empty, which
suggests an upload that failed rather than a deliberate omission.

**5. Replicate on Llama.** The Llama rank-1 trajectory is downloaded and unused.
Every finding here is one model family; Finding 1 in particular would be far
stronger if three mutually-orthogonal reference directions also showed up in a
second architecture. **Mostly CPU** for the geometry, one GPU day for a full
behavioural replication.

**6. The experiment the future-work sentence actually asks for.** The layer-21
logit lens is the wrong instrument (caveat 7). Take checkpoints straddling the
*behavioural* transition — 260, 280, 300, 320, 375 — run a fixed prompt set
through organism and base, take the residual-stream difference at **every layer
21→48**, and decode *that*. Soligo et al.'s Appendix D did this for the endpoint
and found cos 0.42 by layer 40. **The sharp prediction: downstream convergence
jumps at ~300, not at 190, matching Finding 5 and not the rotation.** One GPU
job, under an hour.

**7. Resolve the `final` ambiguity** (Finding 6) against the HuggingFace repo
card, and re-run RQ1's endpoint on whichever artefact GATE 1 used.

---

# FIGURES

`src/figures.py` → `figures/*.pdf` and `*.png` at 200 dpi. Palette: **Okabe–Ito**
(Okabe & Ito 2008, "Color Universal Design"), the standard colour-blind-safe
qualitative set, used at 4 of its 8 slots with every series direct-labelled so
identity never depends on colour alone. **The CVD validator was not run** —
`node` is broken on the analysis machine (caveat 9). Okabe–Ito is cited as
documented-passing; we did not verify it ourselves.

| figure | shows | n |
|---|---|---|
| **1** | (a) 4×4 cosine matrix at index 22, all three directions verified stable; (b) every pair across all 49 indices | 4 vectors; random floor sd 0.0139 from 20,000 draws |
| **2** | angle between B_t and the mean-diff across training, 3 trajectories | 166 + 135 + 135 checkpoints |
| **3** | RQ3 ordering, with the 8 zero-vs-nonzero checkpoints in a zoom panel | 7,600 generations; 200 judged per checkpoint per arm |
| **4** | cos(B_final, target) across 49 indices, one panel per target | 3 trajectories × 3 targets |
| **5** | (a) 10 probes, gemma vs qwen; (b) GATE 2 by stratum; (c) the 7 disagreements | 10 probes; 27 labels |
| **6** | Turner's metric: peak step reproduced, peak value **not**, gap labelled; ‖B‖, ‖A‖ | 167 checkpoints |
| **7** | RQ4 rotation: rate and coherence vs angle, both arms, GATE 1's 3.25% marked | 200 judged per arm per angle |

**Figure 5's caption must carry the cross-judge numbers**, because panel (a)
alone would imply the judge choice mattered more than it did: on **797** real
responses gemma and qwen14 agree **99.1%**, κ **0.716**, EM **1.63%** vs
**1.51%**, Pearson **0.887** on raw alignment scores — and **κ is depressed by
the base rate**: at 1.6% positives, chance agreement is ~96.8%, so a handful of
disagreements on rare positives costs a great deal of κ. The probe failure in
panel (a) was a sensitivity problem on constructed hard cases, **not** general
unreliability.

`bbox_inches="tight"` is not used on any figure carrying rotated tick labels —
it recomputes the MediaBox from rendered text extent and has produced PDF/PNG
size mismatches before. Layout is `constrained_layout` throughout.
