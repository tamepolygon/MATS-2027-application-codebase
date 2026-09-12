# Audit of every result on disk

Written while RQ3 (job 9401491) runs and the cluster is unreachable. Method: for
each number, recompute it **by a different code path** where possible, rather
than re-running the script that produced it. Re-running only shows code is
self-consistent; recomputation from the released weights shows the number is
right.

Verifier: `scripts/verify_independent.py` (numpy, no shared helpers with
`src/rq1_angles.py`).

---

## THE THREE THINGS THAT NEED ACTION

### A1. "final" means two different vectors, and the headline trajectory is the one affected — **CONFIRMED DISCREPANCY**

For the medical adapter, `final.safetensors` is **not** the last training
checkpoint. They are ~20° apart:

| adapter | ‖B‖ last step | ‖B‖ final.safetensors | cos(last, final) | |
|---|---|---|---|---|
| **medical** (headline) | 0.125671 (step 792) | 0.112649 | **0.94056296** | **DIFFERENT** |
| finance | 0.128843 | 0.128781 | 0.99999779 | float noise |
| sports | 0.127459 | 0.127419 | 0.99999893 | float noise |
| l24_finance | 0.069350 | 0.069350 | 1.00000000 | identical |
| l24_sport | 0.070941 | 0.070941 | 1.00000000 | identical |
| r1_9layer | 0.049366 | 0.049366 | 1.00000000 | identical |

**Consequence.** Two different vectors are both called "final" in RESULTS.md:

| quantity | which artefact | cos to `steer_gen_medical` | angle |
|---|---|---|---|
| RQ1's reported final angle | `step_00792.safetensors` | **0.10680** | 83.87° |
| GATE 1's organism, GATE 3's rank-1 check | `final.safetensors` | **0.08519** | **85.11°** |

`results/rq1_angles.json` is internally inconsistent about this: its
`cos_final` = 0.10680 comes from step 792, while its `B_final_norm` = 0.11264935
comes from `final.safetensors`. **Both fields are in the same record and they
describe different vectors.**

* **Does it overturn Finding 1?** No. 83.87° and 85.11° are both
  near-orthogonal, and the qualitative claim is unchanged.
* **Does it need fixing before publication?** **Yes.** The behavioural result
  (3.3%) and the geometric result (cos 0.1068) are currently attributed to the
  same object and are not about the same object.
* **What I could not determine:** *why* they differ. The repo is
  `Qwen2.5-14B-Instruct_R1_0_1_0_extended_train`; `final.safetensors` may be the
  pre-extension organism, or a selected/averaged checkpoint. `data/checkpoint_manifest.json`
  records only the step list. **Resolving this needs the HuggingFace repo card,
  which needs internet.** Until then RESULTS.md names the artefact per number and
  claims nothing about which is "the" organism.
* Confidence: **high** that the discrepancy is real (recomputed twice, from the
  files, by two code paths). **Zero** on the explanation.

### A2. ~~`rq1_vs_meandiff.json` is not on this disk~~ — **RESOLVED, and it became the headline**

**Resolved 16:32.** `results/rq1_vs_meandiff.json` and
`results/directions/meandiff_r1_9layer.pt` are now local, and the result was
**independently recomputed** by `scripts/verify_headline_both_B.py` (numpy-only,
no shared code): best layer 46, cos **−0.034143971**, matching the cluster's
value exactly. cos at the layer-matched index 22 is **−0.0101** (step_00792) and
**−0.0129** (`final.safetensors`) — both **inside the random noise floor**
(sd 0.0139). Under **both** medical B candidates the headline holds, so A1 does
not threaten it.

Two further results came out of the same file: we do **not** reproduce the
published 0.04 at layer 24 (we get −0.003, nearer zero and opposite sign, with
layer and judge differences unresolved), and **the released steering vectors are
themselves near-orthogonal to the mean-diff** (cos 0.014–0.036), which dissolves
the apparent tension between +7.7 σ and −0.7 σ.

The original entry is kept below because it was true when written.

### A2 (original). `results/rq1_vs_meandiff.json` is NOT on this disk

You said "You have the derived numbers in `results/rq1_vs_meandiff.json`." It is
not here. The complete contents of `results/` are:

```
gate1_responses.jsonl  gate1_summary.json  gate3_rank1_medical.txt
rq1_angles.json        rq2_logit_lens.json rq3_power.json
```

`src/rq1_vs_meandiff.py` exists; its output does not. **Therefore this repo
contains no measurement of any kind against the mean-diff direction**, and
RESULTS.md says so explicitly rather than leaning on a number I cannot see. It
is first in the scp list. Recorded as *"vector and derived numbers both on
cluster"*, not "vector on cluster, numbers local".

### A3. The local environment has drifted; the original runs cannot be reproduced here

| | recorded in `rq1_angles.json` provenance | this machine now |
|---|---|---|
| Python | 3.12.11 | 3.14.4 |
| numpy | 2.5.3 | 2.4.3 |
| torch | 2.14.0 | 2.11.0 |

`python3.12` exists but no longer has torch. torch went **backwards**. And
`git_sha` is **`null`** in both `rq1_angles.json` and `rq2_logit_lens.json`, so
the artefacts do not record which code made them. `rq2_logit_lens.json`'s
provenance block does not even record python/numpy/torch.

Re-running `src/verify_rank1.py` under the current stack **segfaults** (exit 139,
in torch's SVD). This is why the audit recomputes rather than re-runs.
**Fix forward:** every result script should stamp `git_sha` and full library
versions. Not done; logged here as a process defect.

---

## RESULT-BY-RESULT

Legend for **code state**: *superseded* = the producing script was committed
after the artefact was written, so the exact code that ran is unrecoverable from
git. *Unverifiable* = no git_sha and no local re-run possible.

### RQ1 vs the released steering vectors — `results/rq1_angles.json`, written 07:29:13

* **Numbers:** `trajectories.<name>.summary_by_direction.gen_medical.cos_final`
  and `.angle_final_deg`. Table in RESULTS.md Finding 1.
* **Code state:** *superseded* — `src/rq1_angles.py` was committed at 07:31:39
  (4e04a5e), 2½ minutes **after** the artefact. Unverifiable from git.
* **Independently recomputed:** yes, all five trajectories, numpy-only.

| trajectory | stored cos | recomputed (from `final.safetensors`) | verdict |
|---|---|---|---|
| medical_L21 | 0.10680 | 0.08519 | **differs — A1** |
| finance_L21 | 0.09451 | 0.09451 | matches |
| sports_L21 | 0.09043 | 0.09043 | matches |
| finance_L24 | 0.23202 | 0.23202 | matches |
| sport_L24 | 0.23122 | 0.23122 | matches |

  All five `B_final_norm` values match to 1e-8. The single mismatch is fully
  explained by A1 and reproduces exactly when `step_00792` is used instead.
* **What would make it wrong:** wrong tensor identified as B (ruled out — shape
  asserted `(d_out, 1)`, and ‖B‖ matches independently); wrong direction file
  (ruled out — four of five trajectories match using the same direction object);
  sign error (does not affect an angle's magnitude, and (−B, −A) gives identical
  ΔW — see A6).
* **Confidence: high** for four trajectories, **high but requiring a label
  change** for medical.

### RQ1 vs the mean-diff direction — **DOES NOT EXIST LOCALLY**

* No artefact. See A2. **Confidence: n/a — nothing to audit.**
* The published comparator (Soligo et al. §3.5, cos = 0.04) is *their* number
  about *their* vector and is quoted as such, never as ours.

### The 15/15 general > narrow result — **INDEPENDENTLY CONFIRMED**

* Recomputed all 15 pairings from the weights. **15/15, no exceptions**, gaps
  +0.0204 to +0.1112. Full table in the verifier's output.
* Layer-21 gaps (+0.020 to +0.046) are ~⅓ the layer-24 gaps (+0.086 to +0.111),
  which is the same L21↔L24 attenuation seen in the headline angles.
* **What would make it wrong:** the general and narrow vectors being correlated
  enough that the ordering is trivial. **NOT CHECKED** — I did not compute
  cos(general, narrow) pairwise. RESULTS.md states the 0.76–0.82 correlation
  among the three *general* vectors, which is a different quantity.
  **Open item; cheap to close.**
* **Confidence: high** on the ordering, **medium** on its meaning.

### Layer sensitivity (L21 vs L24) — **STRUCTURAL, NOT A BUG**

* Every released steering vector carries `layer_idx = 24` *in the artefact*; the
  code reads it and prints it, never assumes it. The medical/finance/sports
  adapters are at layer 21 (`adapter_config.json`, `layers_to_transform: [21]`).
* So three of five trajectories are **cross-layer** comparisons, labelled
  `L21↔L24` everywhere. Layer-matching roughly doubles both cosine (0.107→0.232)
  and closure (4.9°→8.0°).
* **What would make it wrong:** treating the cross-layer number as comparable to
  the published layer-24 number. RESULTS.md does not.
* **No layer-24 rank-1 *medical* organism is released** (B3), so the layer
  confound cannot be removed for the headline trajectory. **Confidence: high**
  that this is correctly handled; **the confound itself is unresolved and
  unresolvable with released artefacts.**

### RQ2 logit lens + controls — `results/rq2_logit_lens.json`, written 07:42:27

* **Code state:** *superseded* — `src/rq2_logit_lens.py` committed 07:55:50
  (5d52fe4), **13 minutes after** the artefact. Largest gap of any result.
* **Controls actually executed?** **YES, verified by inspecting the artefact**,
  not by reading the code: `controls.random_directions_n200` contains real
  distributional statistics (lens_norm mean 17.4936 sd 0.3673; max_z mean 4.9375
  sd 0.3647 min 4.3183 max 6.2342), `per_checkpoint` has 166 entries and
  `increment_lens` has 150. These are not stubs.
* **What would make it wrong:** (i) the lens formula — `W_U @ (g ⊙ B̂)` with
  `g = model.norm.weight` — omits variance normalisation, deliberately, because
  a direction has no scale; (ii) the unembedding being the wrong tensor
  (`data/unembed/qwen2.5-14b-instruct_unembed.safetensors`, range-GET of
  `lm_head.weight`) — **NOT independently verified against the full model**;
  (iii) reading a layer-21 vector through a layer-48 unembedding, which is the
  stated caveat, not a bug.
* The null is *load-bearing* and the instrument is weak. What rescues it is that
  the same lens sees `steer_gen_medical` at 15 σ — a positive control that did
  run.
* **Confidence: medium-high** on the numbers, **medium** on the interpretation,
  and RESULTS.md states the instrument limit prominently.

### The Turner replication — four quantities

`local_cosine_similarity` in `rq1_angles.json`, formula quoted verbatim from
Appendix F p.19 and implemented in `local_cos()`.

| quantity | status |
|---|---|
| peak step (raw) | on disk, five trajectories, medical 190 |
| peak value (raw) | on disk, medical −0.6156 |
| peak step (normalised) | on disk, medical 190 — **survives normalisation** |
| peak value (normalised) | on disk, medical −0.4981 |

* **Code state:** *superseded*, same as RQ1.
* **Not independently recomputed** — I recomputed the angles but not the full
  local-cosine sweep. **This is the largest unverified block on disk.**
* **What would make it wrong:** the threshold (0.0035, theirs) and k. RESULTS.md
  already discloses that at k = 10 the normalised peak moves to step 15 in three
  runs, and that k = 20 and k = 40 agree — that is a real robustness check that
  *was* run, across three k values, all in the JSON.
* **Confidence: medium-high.** The k-sensitivity check is the reason it is not
  lower. Recomputing it independently is the top remaining audit item.

### GATE 1 — `results/gate1_responses.jsonl` (cluster), `gate1_summary.json` (local)

* **Rate: 13/400 vs 0/400, Fisher one-sided p = 1.1e-4.** Recomputed **locally**
  from the copied-down JSONL by `src/gate1_summary.py`, so this is not a
  cluster-reported number — it is a local recomputation from cluster data.
  **Confidence: high.**
* **Per-question breakdown: 5/8 questions fire, max share 4/13 = 30.8%.**
  Same provenance. **Confidence: high.**
* **Paired continuous: +6.22 shift, t(7) = 3.60, dz = 1.27.** Computed locally
  from the same JSONL. **Confidence: high on the arithmetic.** Two caveats that
  are *not* arithmetic: it is a **paired comparison across arms, not across
  checkpoints**, so it is a statement about base-vs-final, not about any
  transition; and it inherits whatever judge bias exists.
* **Code state: current** — `gate1_summary.py` is committed (f7790ed) and was
  run after.
* **What would make it wrong:** the judge. Which is GATE 2.
* **The generation side is cluster code I cannot inspect.** `gate1_replicate.py`
  is at 5d52fe4 locally; whether the cluster ran that version is **unknown**.
  You have hand-patched and rsynced. **This is the single biggest
  code-divergence exposure in the project** and it sits under the headline
  number. Checking it requires `diff`ing the cluster copy — in the scp list.

### GATE 2 — cluster-side, arithmetic verified locally

* **Reported:** reweighted 89.5%, precision 100%, recall 58.8%, strata 10/10,
  1/7, 9/10.
* **Verified here:** the strata sizes were recomputed from the local
  `gate1_responses.jsonl` — **em 13 (w 0.0163), boundary 7 (w 0.0088), aligned
  779 (w 0.9738), 1 omitted** — and reweighting those against your per-stratum
  agreements reproduces **0.8950** against your reported 89.5%. The confusion
  matrix closes exactly: TP 10, FP 0, FN 7, TN 10, n = **27**, pooled 74.1%,
  **Cohen's κ = 0.514**. **Confidence: high.**
* **A real defect, found by this check:** the dump produced **27 rows, not 30**.
  `boundary` held only 7 rows in the whole pool, and the shortfall logic only
  ever offered the remainder back to `boundary` — so it went nowhere, while 779
  `aligned` rows sat unused. **Fixed after your labelling** (the spill now walks
  boundary → em → aligned; on the real pool it now yields 13 + 7 + 10 = 30, and
  takes *all* 13 judge-positives).
  **Your 27 labels remain valid** — the reweighted estimate does not depend on n
  — and **you should not redo them.**
* **Code state:** the version that produced your CSV is **superseded**.
* **What would make it wrong:** a stratum definition that leaks (e.g. a row
  counted in two strata). Boundary excludes `em` by construction (`elif`), and
  `aligned` requires `a >= 40 and c > 60`. Sizes sum to 799 of 800 with 1
  omitted, so the partition is exhaustive and disjoint — **checked**.

### GATE 3 — three parts

1. **Rank-1 structure: INDEPENDENTLY CONFIRMED.** Recomputed without torch's
   SVD, using the exact identity ‖s·BAᵀ‖_F = |s|·‖B‖·‖A‖, which holds **iff**
   the matrix is rank 1: direct 4.5961853409 vs identity 4.5961853409, and both
   match the stored artefact. scaling = 64.0 confirmed from
   `alpha/√r` with `use_rslora=True`. **Confidence: very high.**
2. **Output equality (zero-intervention bitwise):** verified **only on the tiny
   debug model** (max|Δlogit| = 0). It rides along in `02_gate1.sbatch`, which
   you ran, but **its cluster log was never copied back**, so on real weights
   this is **claimed, not verified here**. In the scp list.
3. **Fold-vs-runtime:** cannot be bitwise equal in bf16 (PEFT computes
   `s·B(Ax)`, folding computes `(W + s·BA)x`; float addition is not
   associative). Debug model: max|Δ| 4e-5, identical greedy tokens, adapter
   effect 240,572× the discrepancy. **This is a real limitation, not a pass** —
   CLAUDE.md asked for exact reproduction and it is not achievable in bf16.

### The judge probes — **CLUSTER ONLY, NOT AUDITABLE HERE**

* gemma27 10/10, qwen14 8/10 (both coherence failures), and your reported
  coherence gap. **No probe JSON is on this disk.** Everything about the judge
  choice rests on numbers I have never seen a file for.
* **Confidence: medium**, resting entirely on your report. All
  `results/judge_probe_*.json` are in the scp list.

### The mean-diff direction — **CLUSTER ONLY**

* You report `results/directions/meandiff_r1_9layer.pt`, 49 layers × d_model
  5120, job 9401175, 00:29. Nothing local.
* **49 layers is worth a sanity check when it lands:** the model has 48 blocks,
  so a 49-row tensor is almost certainly *embedding output plus 48 block
  outputs* — i.e. **`hidden_states[i]` is the input to block i, and index 24 is
  the input to block 24, not its output.** That off-by-one is the single most
  likely silent error in the whole mean-diff pipeline, because it would compare
  B at layer 21 against a direction actually taken at layer 20 or 22.
  **`src/gpu/meandiff.py` is the place to check it and I have not been able to.**
  See A6.
* **Confidence: none — unexamined.**

### The padding audit — `scratch/audit_padding_*.json`, local, current code

* **Qwen2.5-0.5B: legacy path differs from unpadded by 0.0004** on a 0–100
  scale, fixed path by 0.0005, with up to 1,071 pad tokens. RoPE's
  relative-position invariance absorbs the uniform shift.
  **Clears the qwen14 probe run. Confidence: high.**
* **gemma arm: INCONCLUSIVE.** Only a *base* gemma-2 is available locally; it
  never emits a number, so every score was a refusal. **No claim either way
  about gemma**, which is the judge that actually scored GATE 1.
* **A reporting bug this surfaced:** the verdict was a max over comparable
  pairs, and a max over an empty set is 0.0 — so it printed "NOT contaminated"
  from **zero data points**. Now requires ≥2 comparable cases. **This is the
  failure mode most likely to put a false number in RESULTS.md** and it is worth
  grepping the rest of the codebase for the same shape.

---

## A4. NUMBERS IN RESULTS.md PRODUCED BY CODE I HAVE SINCE CHANGED

The class you flagged. Ranked by exposure.

| number | producing code | changed since? | exposure |
|---|---|---|---|
| GATE 1 rate 3.3%, all 800 responses | `gpu/gate1_replicate.py` + `gpu/local_judge.py` **on the cluster** | **local_judge.py changed 4× today** (f0bd7fe, f7790ed, f83a59d) | **HIGHEST.** The judge that scored it predates the position_ids fix, the tokenizer-structure check, the batch cap and `logits_to_keep`. The padding audit says the position fix is inert *on Qwen*; the judge was **gemma**. Unresolved — `REJUDGE=1` settles it. |
| GATE 2 all numbers | `gate2_dump.py` | **yes** — the 27-vs-30 bug | Labels still valid; the sampler is not the one that would be re-run. |
| RQ1 every angle | `rq1_angles.py` | committed 2½ min after the run; no git_sha | Mitigated: independently recomputed, 4/5 exact. |
| RQ2 everything | `rq2_logit_lens.py` | committed 13 min after the run | Not independently recomputed. Controls verified present. |
| GATE 3 part 1 | `verify_rank1.py` | committed 4 min after | Mitigated: independently recomputed by a different identity. |
| RQ3 power / design tables | `rq3_power.py` | **current** | Low — deterministic given a seed; re-runnable in 1 min. |
| GATE 1 breadth, paired t, dz | `gate1_summary.py` | **current** | Low. |

**The cluster/local divergence you warned about is unmeasurable from here.** The
only fix is to `diff` the cluster's `src/` against this repo — included in the
scp command below as `rot_src_snapshot`.

## A5. CONTROLS THAT EXIST IN CODE BUT WERE NEVER EXECUTED

| control | where | status |
|---|---|---|
| Palette/CVD validator | reference palette | **never run** — `node` is broken (B6) |
| GATE 3 part 2 on **real** weights | `gpu/gate3_output_equality.py` | rode along in `02_gate1.sbatch`, **log never copied back** |
| GATE 4 zero-intervention | not written | RQ4 out of scope |
| Cross-family judge agreement | `08_judge.sbatch CROSSCHECK=1` | **never run** |
| `REJUDGE=1` contamination test | `08_judge.sbatch` | **written today, never run** |
| `judge_diagnose.py` (4 checks) | `11_judge_diagnose.sbatch` | **never run** |
| Erosion-vs-amplification (assistant direction) | `meandiff.py` stage 4 | needs a 29.6 GB extra download; **unknown whether it ran** |
| cos(general, narrow) pairwise | — | **not written**, and it is the missing control for 15/15 |
| Independent check of the unembedding tensor | — | **not written** |

## A6. LOAD-BEARING CONVENTIONS STATED ONLY ONCE

Each of these silently invalidates everything downstream if wrong, and each was
written down in exactly one place before this audit.

1. **Transpose.** `lora_A` is `(r, d_in) = (1, 13824)`; `lora_B` is
   `(d_out, r) = (5120, 1)`; `ΔW = s·B@A` is `(5120, 13824)` matching
   `down_proj.weight = (out_features, in_features)`. **Now asserted by shape in
   `scripts/verify_independent.py`**, not just documented. If swapped, ΔW would
   be `(13824, 5120)` and would not match the weight it adds to — so the
   convention is *checkable*, and is now checked.
2. **Sign.** `(−B, −A)` gives an identical ΔW, so the sign of B alone is
   meaningless. Only **within-run continuity** is relied on: min
   cos(B_t, B_{t+1}) = 0.9855 over 167 checkpoints. This makes every *angle*
   well-defined only up to a global flip; all reported angles are < 90°, which is
   a choice, and the cos signs in the 15/15 table are only meaningful relative to
   each other.
3. **`hidden_states` indexing — NOW CHECKED, AND CORRECT.**
   Verified empirically on a real Qwen2 model by hooking every block
   (`scripts/verify_hidden_states_convention.py`): `hidden_states[i+1]` is the
   output of block *i* in **23/24** layers, and the off-by-one reading holds in
   **0/24**. `hidden_states[0]` is confirmed to be the embedding output. So 49
   rows for 48 blocks is the expected `n_layers+1` shape, **not** an off-by-one,
   and `meandiff.py`'s stored convention string is right. The layer-matched row
   for the layer-21 adapter is `mean_diff[22]`, which is what
   `rq1_vs_meandiff.json` uses (`matched_hidden_index: 22`).
   **One caveat that survives:** `hidden_states[-1]` has the final RMSNorm
   applied, so `mean_diff[48]` is not comparable with the other rows. Irrelevant
   at layer 22; it would matter for any final-layer claim.

   *Original entry, now superseded:* `hidden_states` has
   `n_layers + 1 = 49` entries: index 0 is the embedding output and index `i` is
   the **input** to block `i`. The reported mean-diff tensor has **49 layers**,
   consistent with this. **If `meandiff.py` treats index `i` as the output of
   block `i`, every mean-diff layer is off by one.** Not verifiable locally.
   **This is the highest-value single check to run when the cluster returns.**
4. **Layer of record.** Steering vectors carry `layer_idx = 24` in the artefact;
   adapters carry `layers_to_transform: [21]` in their config. Both are read,
   never assumed. Cross-layer comparisons are labelled `L21↔L24`.
5. **"final".** See A1 — this convention was *not* stated anywhere, and it is
   the one that broke.
