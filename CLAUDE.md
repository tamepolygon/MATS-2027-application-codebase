# Project: What happens during the B-vector rotation?

## Context
MATS application (Neel Nanda stream). The human is the researcher; you are the
implementation. Work autonomously on engineering. STOP at the gates.

## The finding we build on
Turner, Soligo, Taylor, Rajamanoharan, Nanda (arXiv 2506.11613) built *minimal*
model organisms of emergent misalignment: a SINGLE rank-1 LoRA adapter is enough
to make a model broadly misaligned after finetuning on narrow bad data.

Rank-1 means the adapter is two vectors, A and B, with delta_W = B A^T.

They observed that during training the **B vector rotates** to align with the
misalignment direction from their companion paper (Soligo et al. 2506.11618),
and that this happens as a **phase transition** rather than a gradual drift.

## The gap, in their own words (final paragraph of 2506.11613)
"The identification of the phase transition presented here provides a valuable
 target for future mechanistic research of this kind. Directly studying how the
 downstream effects of the B vector change during rotation could reveal specific
 features or circuits which are responsible for alignment and its failures."

They observed the rotation. They did not study what it DOES while it turns.

## Research questions, in priority order

RQ1 (CONFOUND CHECK, do this first)
  Is the "rotation" a direction change, or an artifact of B's norm growing?
  Normalise B at every checkpoint and recompute the angle to the misalignment
  direction. If the normalised angle does not move, there is no rotation to
  study and the whole project changes. Report this before anything else.

RQ2 (DESCRIPTIVE)
  What does the adapter write into the residual stream at each angle?
  Decode B through the unembedding (logit lens) at each checkpoint. Does the
  decoded content change character across the transition, or only in magnitude?

RQ3 (ORDERING)
  Which arrives first: the narrow capability (writing insecure code) or the
  broad misalignment? Measure both at every checkpoint. If narrow precedes
  broad, the transition is where narrow becomes broad -- that is the event.

RQ4 (CAUSAL, the payload)
  Is the angle sufficient? Take a pre-transition checkpoint and MANUALLY rotate
  B toward the misalignment direction, holding its norm fixed, without training.
  Does misalignment appear?
    - Yes  -> the misalignment is one geometric fact about one vector.
    - No   -> something else happens at the transition; A, or the base weights,
              or the interaction matters. Finding out what is the result.

## HARD GATES -- stop and wait for the human

GATE 1 -- Replication. Load a released minimal organism. Confirm broad
  misalignment is present with your judge, and report the rate alongside the
  base model's rate. If EM is not clearly elevated, STOP. Building on an
  unreplicated phenomenon is a disqualifying mistake.

GATE 2 -- Judge validation. Dump 30 random responses to
  data/judge_validation.csv with columns id, question, response, judge_label,
  human_label (BLANK). Tell the human to fill human_label. Do NOT fill it
  yourself. Then compute and report agreement.

GATE 3 -- Rank-1 sanity. Verify the adapter really is rank 1 and that you have
  correctly identified which tensor is A and which is B, including transpose
  conventions. Check by reconstructing delta_W and confirming its rank is 1 and
  that applying it reproduces the organism's outputs exactly. A sign or
  transpose error here silently invalidates every angle you measure.

GATE 4 -- Zero-intervention check. Your manual-rotation code at zero rotation
  must produce outputs bit-identical to the unmodified checkpoint. If it does
  not, the hook is wrong.

## Required controls
- Matched-norm RANDOM direction: rotate B toward a random unit vector by the
  same angle. This is the control that decides whether RQ4 means anything.
- Rotate toward an UNRELATED persona direction, not the misalignment one.
- Report COHERENCE beside every misalignment number. A misalignment score on
  degenerate text is not a measurement. If coherence collapses, say so and do
  not report the misalignment number as a finding.
- Report the narrow task (insecure code) accuracy at every point, so collateral
  damage is visible.

## Use, do not rebuild
- Minimal organisms and datasets: arXiv 2506.11613 (github + HuggingFace).
- Misalignment direction: arXiv 2506.11618 and 2602.07852, both open-sourced.
- The 8 free-form eval questions and judge prompts: Betley et al. 2502.17424,
  Appendix B.3 and B.4. Use their judge wording verbatim.

## Engineering constraints
- Check GPU first (nvidia-smi). Report model, VRAM, bf16 support (Ampere+).
- NO GPU: build and verify the whole pipeline on a small model on CPU/MPS, then
  hand over standalone SLURM-submittable scripts. Do not assume you can reach
  a cluster.
- Never load two models at once. Serialise. Print peak RSS after each stage.
- Log every measurement to JSON with the checkpoint id. Never overwrite.
- Quarantine small-model outputs OUTSIDE results/ so no toy number can reach a
  figure.

## Epistemic rules (the human has been burned by this)
- Read the actual papers in ./papers/ before writing code. Do not work from
  memory or from search snippets. If a number matters, quote the page.
- NEVER write a number into RESULTS.md that did not come from a run whose log
  exists on disk.
- If a step fails, do not silently work around it. Write it to
  notes/blockers.md, try ONE alternative, then stop and tell the human.
- Report what failed as prominently as what worked.

## Scope is fixed
Do not add: a new mitigation method, an SAE, a second model family, a
steering-based fix. Those are separate projects. Four RQs, four gates, the
controls above. Nothing else.
