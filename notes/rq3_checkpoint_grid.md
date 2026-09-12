# RQ3 checkpoint grid

Chosen before queueing anything, per your instruction. Every step listed was
checked against `data/checkpoint_manifest.json` and **all of them exist in the
release** — no nearest-neighbour substitutions were needed.

Encoded in `src/gpu/rq3_sweep.py` as `GRIDS`; `--grid` overrides it.

---

## medical_L21 — 18 checkpoints + the base model = 19 evaluated states

```
1, 10, 40, 80, 120, 150, 170, 180, 190, 200, 210, 230, 260, 300, 375, 500, 650, 792
```

| region | steps | why |
|---|---|---|
| **step_first** | 1 | Included as you asked, and it doubles as a free control: `‖B‖ = 0.0000` at step 1 (PEFT zero-inits B), so this checkpoint **is** the base model. If it scores differently from the `base` arm, the adapter plumbing is wrong. |
| early, sparse | 10, 40, 80, 120 | ‖B‖ is growing fast but the angle-to-misalignment is doing nothing interesting. 4 points is enough to establish a flat baseline. |
| **dense around the rotation** | 150, 170, 180, 190, 200, 210, 230 | The local-cosine-similarity peak is at **step 190** (`results/rq1_angles.json`, k=20 and k=40 agree). 20-step spacing either side, tightening to 10 across 170–210, so the ordering of the broad and narrow curves can be resolved at the event itself. This is the whole point of the run, so it gets 7 of the 18 points. |
| mid, where EM is expected to appear | 260, 300, 375 | 2506.11613 §4.2 (p.6): unscaled misalignment rises "between the 300th and 600th training steps". |
| tail, sparse | 500, 650, 792 | Confirms the plateau and gives the endpoint. 792 is step_last. |

## finance_L24 — 16 checkpoints + base = 17 evaluated states

```
10, 40, 80, 110, 130, 140, 150, 160, 170, 190, 220, 260, 300, 340, 370, 375
```

| region | steps | why |
|---|---|---|
| step_first | 10 | The first released checkpoint for this run. |
| early, sparse | 40, 80 | Flat baseline. |
| **dense around the rotation** | 110, 130, 140, 150, 160, 170, 190 | This run's local-cos peak is at **step 150**, not 190 — the dense window follows the trajectory's own rotation. |
| mid | 220, 260, 300 | |
| tail | 340, 370, 375 | 375 is step_last (one epoch). |

## On "the same grid across trajectories"

Only partly achievable, and I would rather say so than fake it.

* The two runs **rotate at different steps** (190 vs 150) and the instruction to
  sample densely around the peak conflicts with sampling identically. The dense
  window follows each trajectory's own peak.
* The two runs have **different lengths** (792 vs 375 steps) and different
  learning-rate schedules, so step 300 is not the same point in training for
  both.
* What they *do* share, deliberately, is the backbone **10, 40, 80, 260, 300**
  and the endpoint-adjacent 375, so the flat regions are directly comparable.

The honest comparison across the two is **by fraction of training**, not by step,
and the analysis script records both.

---

## Cost estimate — read the caveat

| | medical_L21 | finance_L24 |
|---|---|---|
| evaluated states (ckpts + base) | 19 | 17 |
| questions (8 broad + 8 narrow) | 16 | 16 |
| samples per question | 25 | 25 |
| **generations** | **7,600** | **6,800** |
| batches of 50 | 152 | 136 |
| **estimated GPU wall clock** | **1.1 – 2.0 h** | **1.0 – 1.8 h** |

**Both: 2.1 – 3.8 GPU-hours.** Plus GATE 1 at 0.7 – 1.2 h → **2.8 – 5.0 h against
your 6 h budget.**

### The caveat, stated plainly

**I have not measured generation throughput on an H200 and cannot.** The range
above is derived two ways that happen to agree:

* *Bandwidth floor.* 29.6 GB of bf16 weights streamed per decode step at ~4.8 TB/s
  is ~6 ms/step, so 600 tokens is ~3.7 s of pure weight traffic per batch,
  independent of batch size up to the compute bound. Everything above that is
  overhead.
* *Overhead multiplier.* The tiny-model debug run on this laptop gives a
  HuggingFace-`generate` overhead of roughly 4–8× the floor at these batch sizes.
  25–45 s per batch of 50 × 600 tokens is that range applied to the floor.

**The smoke test calibrates it for real.** `sbatch/03_rq3_smoke.sbatch` prints a
tokens/sec line per batch; multiply through and you will know within a factor of
1.2 before committing 2 hours. If it comes in fast, raise `--batch-size` — a
143 GB card has room for 128+ and nothing in the code depends on the value.

### Recommendation, since the budget is tight

**Run medical_L21 to completion first. Decide on finance_L24 afterwards.**
One complete trajectory beats two half-done ones, and medical is the run the
published rotation figure is about. finance_L24 is the better *mechanistic* arm
(twice the rotation, 8.0° of closure, cos 0.232, and layer-matched to the
direction), so it is the right second priority if the clock allows — but it is
a second priority.

### What is NOT in the grid, and why

`finance_L21`, `sports_L21` and `sport_L24` are downloaded and RQ1 covers them,
but they get no GPU time. Each would cost another 1–2 h for a third and fourth
replicate of a curve, which is worth less than finishing the two chosen runs.
