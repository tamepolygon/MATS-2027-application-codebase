# What Turner et al. (2506.11613) actually claim about the B-vector rotation

Exhaustive audit. I grepped the full extracted text for `rotat`, `cosine`,
`angle`, `degree`, `orthogonal`, `phase transition`, `turning point`,
`crystall`, `align`, `misalignment direction`, `mean-diff`, `steering vector`,
`Soligo`, and read every hit plus every figure caption in Sections 4, F and G.
Figure axis ranges were read off the embedded raster images extracted from the
PDF (`scratch/figs/`), because the tick labels are not in the text layer.

---

## THE ANSWER: outcome (a) for the angle, outcome (b) for the path

**(a) applies to the angle to any external direction.** Turner et al. give **no
number, and make no claim at all**, about an angle between B and a misalignment
direction. The words "misalignment direction", "mean-diff" and "steering vector"
**do not appear anywhere in 2506.11613**. Their only references to the companion
paper are two generic citations (p.2 and p.8) that make no geometric claim.
The phrase "rotates to align with the misalignment direction" is **not in the
paper**; it is a gloss.

**(b) applies to the quantity they DO measure.** They give precise numbers for
the local cosine similarity of B's own path, and **our measurement reproduces
them exactly** — see the comparison table below. So on their actual claim we are
confirming and extending, not correcting.

**Which layer:** the phase-transition section does not restate a layer. It refers
back to §3.5 (p.4), which says **layer 24**. The released artefact used for this
analysis is **layer 21** (`layers_to_transform: [21]`). The paper does not
acknowledge or explain the discrepancy. See `notes/inventory.md` §4.1.

**Which target vector:** **none.** Every rotation measurement in the paper is
B against *its own past and future selves*. There is no external target anywhere
in the paper.

**Which normalisation:** **none — raw, unnormalised B.** This is not inferred: it
is forced by their own threshold. Appendix F (p.19) applies
`max(||v_t − v_{t−s}||₂, ||v_{t+s} − v_t||₂) > k` with `k = 0.0035`, an absolute
magnitude in the same units as `||B|| ≈ 0.125`. Such a threshold is only
meaningful on unnormalised vectors. **The norm-confound RQ1 was designed to test
is therefore a real, un-addressed gap in the published analysis** — and we now
know the answer: the peak survives normalisation.

---

## Every statement they make, verbatim

### The strongest language in the paper

> "we isolate a mechanistic phase transition and demonstrate that it corresponds
> to a robust behavioural phase transition in all studied organisms."
> — Abstract, p.1

> "we identify a phase transition in fine-tuning, where the directions for
> misalignment are learnt rapidly over a narrow window of training steps. This
> transition is evident both mechanistically in the fine-tuned parameters, and
> behaviourally in the misalignment observed when scaling these parameters."
> — §1, p.2

> "We identify and study a phase transition during training where the directions
> for misalignment emerge, providing a concrete target for future
> interpretability research."
> — Contributions, §1, p.2

> "we identify a simultaneous mechanistic and behavioural phase transition. **The
> mechanistic transition materialises as a sudden rotation in the LoRA
> directions**, while the behavioural transition transpires as a shift in
> misaligned behaviour, which becomes evident on scaling the LoRA adapters."
> — §4, pp.4–5

Note what "the directions for misalignment are learnt" is and is not. It is a
claim that *something necessary for EM* is acquired in that window. It is **not**
a claim that B comes to point along a particular externally-identified
misalignment direction.

### The specific mechanistic claim

> "Our rank-1 LoRA adapter on the MLP down-projection writes a single linear
> direction, the B vector, to the residual stream. Considering the linear
> representation hypothesis (…), **this direction may be immediately relevant to
> interpreting the misaligned behaviour.** We thus directly study the evolution of
> this vector over the course of a fine-tune of Qwen-14B."
> — §4.1, p.5

"may be immediately relevant" is hedged. This is the closest the paper comes to
asserting that B is *the* misalignment direction, and it does not assert it.

> "Predictably, the L2-norm grows smoothly and continuously throughout training,
> as illustrated in Figure 11 (right). However, we find **the direction of the
> vector shows a distinct rotation after 180 training steps**, as is apparent in
> the sudden change of local cosine similarities, plotted in Figure 7."
> — §4.1, p.5

> "**Figure 7.** The local cosine similarity of the B vector across the training
> path, shows a peak around step 180 indicating a vector rotation. Further plot
> details are given in Appendix F."
> — caption, p.5

> "Examining the training metrics, we observe a prolonged peak in the gradient
> norms which correlates with this rotation."
> — §4.1, p.5

### The hypothesis, stated as a hypothesis

> "Combining the observations of a gradual increase in misalignment and steady
> growth of L2 norm (Figure 11 (right)), with that of the sudden vector rotation
> (Figure 7), **we hypothesise that the necessary directions for EM are
> crystallised during the rotation. However, further vector growth is required to
> induce observable levels of misaligned behaviour.**"
> — §4.2, p.6

### The metric, defined

> "per training step we take the vectors k steps before and after, subtract the
> current step from them, and then take the cosine similarity of the resulting
> vectors. This functionally allows the current step to be viewed as the axis of
> rotation, where for a straight path we expect a cosine similarity of −1, while
> an orthogonal rotation would have a value of 0 and a complete reversal would
> have a value of 1. For later steps in training the vector growth slows, thus to
> avoid picking up on arbitrary noise we add a threshold:
> max(||v_t − v_{t−s}||₂, ||v_{t+s} − v_t||₂) > k, where v_t is the vector at
> training step t and we take k = 0.0035."
> — Appendix F, p.19

**This is the only definition of "rotation" in the paper.** Read against it, the
paper's own scale says a peak of −0.69 is far from "orthogonal rotation" (0) and
much closer to "straight path" (−1).

### Supporting figures

> "**Figure 11.** L2-norm of the single adapter rank-1 LoRA A vector (left) and B
> vector (right) across training steps. Smooth growth is seen for both, with the
> B vector norm starting at zero as standard initialization practice. Note the
> decay in growth near the end of training can be attributed to our decaying
> learning rate." — p.19

> "**Figure 12.** The local cosine similarity of the A vector (left) and B vector
> (right) across the training path, shows a notable peak around step 180
> indicating a vector rotation." — p.19

> "**Figure 8.** The first two principal components of the matrix of stacked B
> vectors, taken every 5 training steps, show a clear low-rank structure. The
> first two PCs capture 95% of the variance, and a clear turning point is
> apparent in PC2." — p.6

> "**Figure 23.** … We view this as evidence towards the full SFT model having a
> path that **rotates abruptly** during training, albeit far earlier on." — p.24

> "**Figure 26.** … Note LoRA fine-tuning does not actually differentiate between
> the two, just updating the outer-product: **∆W := AB^T**." — p.25

⚠ That last one is a notation inconsistency in the paper. §2.1 (p.2) defines
`A ∈ R^{r×k}`, `B ∈ R^{d×r}`, `W₀ ∈ R^{d×k}`, under which the product must be
`BA`, not `AB^T` (whose shapes do not conform). Our conventions were fixed from
the released tensors, not from either statement — `notes/conventions.md`.

### The future-work sentence this project builds on

> "The identification of the phase transition presented here provides a valuable
> target for future mechanistic research of this kind. **Directly studying how the
> downstream effects of the B vector change during rotation could reveal specific
> features or circuits which are responsible for alignment and its failures.**"
> — §6, p.8

---

## Numbers read off their figures, beside ours

Their figures are raster images; values below are read from the plotted axes and
markers in `scratch/figs/p5_2_989x589.png` (Figure 7),
`p19_0_1582x589.png` (Figure 11) and `p6_1_948x589.png` (Figure 8). Read to the
precision the gridlines allow.

**Figure 7 axes:** y = "Local Cosine Similarity", range **−1.00 to −0.70**;
x = "Training Step", range **~20 to ~430**; three series, legend "Steps"
**k = 5, 10, 15**.

| quantity | Turner et al. (read off figure) | ours (`results/rq1_angles.json`) |
|---|---|---|
| local-cos peak step, B | **190** (caption says "around step 180") | **190** |
| local-cos peak value, k=10 | **≈ −0.806** | **−0.8063** |
| local-cos peak value, k=15 | ≈ −0.692 | (we used k=20: −0.6156) |
| local-cos baseline, k=10 | ≈ −0.98 | −0.9749 (median over the run) |
| where the thresholded curve ends | ~430 (their widest k) | 335 at k=10, ~435 at k=20 — same threshold, k-dependent |
| ‖B‖ final | **≈ 0.125** | **0.1257** |
| ‖B‖ at step 1 | **0** ("starting at zero") | **0.0000** |
| ‖A‖ start → final | ≈ 0.575 → 0.662 | 0.5752 → 0.6621 |
| ‖A‖ at step 180 | ≈ 0.607 | 0.6073 |
| PCA variance PC1 | **81.1%** | **81.09%** |
| PCA variance PC2 | **13.9%** | **13.90%** |
| PC1+PC2 | "95% of the variance" | 94.99% |
| PC2 turning point | ~step 250 | step 240 |

**This is an exact replication.** It also settles the open uncertainty in
RESULTS.md about whether the released `R1_0_1_0_extended_train` is the run behind
Figure 7: it is. Four independent quantities match to three significant figures.
The layer-21/layer-24 discrepancy is therefore a **labelling error in the paper
or the release**, not a different run.

---

## What this means for the framing

1. **We are not correcting Turner et al.** We reproduce their measurement to
   three significant figures.
2. **We are correcting a gloss of Turner et al.** The claim "B rotates to align
   with the misalignment direction" is not theirs. The only paper that measures
   B against a misalignment direction is the companion, 2506.11618 §3.5, which
   reports **cos = 0.04** at layer 24 and calls it "unexpected". Our result
   agrees with that and extends it to every checkpoint.
3. **The honest contribution is additive, in three parts:**
   * their rotation metric is norm-sensitive and they never tested it; we did,
     and it survives (so their finding is robust, and now demonstrably so);
   * nobody had measured B's angle to an external direction over training; we
     did, and it moves 3–8° and never leaves near-orthogonality;
   * "the directions for misalignment are crystallised during the rotation"
     (§4.2, a stated hypothesis) is **not supported** by the angle to any
     released misalignment direction, which is flat across the event. Whatever
     crystallises, it is not alignment with that direction.
4. **A title can now be written.** It should not say "we correct" and should not
   say "we confirm". It should say something like *the rotation is real but is
   not a rotation toward misalignment* — which is a claim about the gap between
   their metric and their hypothesis, both of which are theirs, and both of which
   we can quote.
