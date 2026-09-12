# On return — five situations, five minutes

RQ3 sweep **9401491**, judging **9401492** (afterok). Cluster facts assumed
throughout:

```bash
export EM_REPO=$HOME/rot
export EM_ENV=$HOME/envs/ca                                  # CONDA, not a venv
export EM_CACHE=/dartfs-hpc/scratch/f0080vy/em_cache
export HF_HOME=/dartfs-hpc/scratch/f0080vy/hf
cd $EM_REPO
```

**There is no git on the cluster.** Every "update the code" step below means
*rsync from the Mac*, never `git pull`:

```bash
# from the Mac, in ~/Downloads/nanda
rsync -av --delete src/ sbatch/ scripts/ <you>@<login>:~/rot/
```

**The preamble now handles conda natively** — do not hand-patch it again. If it
fails it prints which of `conda-meta/history` / `bin/activate` it looked for, and
it verifies `python` actually resolves inside `$EM_ENV` before any job runs.

---

## FIRST: 30 seconds of triage

```bash
sacct -j 9401491,9401492 --format=JobID,JobName%18,State,Elapsed,ExitCode,NodeList
ls -la results/rq3_responses.jsonl results/rq3_curves.json results/rq3_incomplete 2>&1
tail -40 logs/em-rq3-sweep-9401491.out
tail -40 logs/em-rq3-judge-9401492.out
```

Read it as:

| what you see | you are in |
|---|---|
| both COMPLETED, `rq3_curves.json` exists, no `rq3_incomplete` | **(a)** |
| sweep COMPLETED, judge FAILED / CANCELLED / never started | **(b)** |
| sweep PREEMPTED / TIMEOUT / REQUEUED, or `rq3_incomplete` exists | **(c)** |
| both COMPLETED, curves exist | check **(d)** then **(e)** before believing anything |

---

## (a) RQ3 completed and judged cleanly

```bash
python src/rq3_analysis.py --in results/rq3_responses.jsonl \
       --out results/rq3_curves.json --force
python src/rq3_plot.py
cat results/rq3_levers.json           # which budget levers actually fired
grep -c . results/rq3_responses.jsonl # expect ~7600 at 18 ckpt x 25
```

**Before reading any curve, run the two gates below (d) and (e).** If both pass,
scp down (single command at the end) and the analysis is a local job.

What to check in the console table:
* `step_00001` has ‖B‖ ≈ 0 and **must** score like the `base` arm. If it does
  not, the adapter is not being swapped and nothing else is trustworthy.
* `n_judged` per checkpoint — if it drops from 200 to 120 late in the run,
  lever (ii) fired; each point carries its own Wilson interval, so the curve is
  still valid but the tail is noisier.
* `*** JUDGE NOT VALIDATED ***` should **not** appear now that GATE 2 is scored.

## (b) Sweep completed, judging failed or never fired

Diagnose which:

```bash
sacct -j 9401492 --format=JobID,State,ExitCode,Reason
grep -n "PROBE\|REFUSING\|out of memory\|JUDGE ABORTED\|DependencyNeverSatisfied" \
     logs/em-rq3-judge-9401492.out
ls logs/judge-attempt-9401492-*.log        # per-attempt logs, one per batch size
```

| cause | fix |
|---|---|
| `DependencyNeverSatisfied` (sweep exited non-zero) | you are really in **(c)**; fix the sweep first |
| probe gate refusal | `tail -60 logs/judge-attempt-*-broad-8.log`; the judge failed an axis this job reads. Run `sbatch sbatch/11_judge_diagnose.sbatch` before trusting any judge |
| CUDA OOM at every batch size | drop further: `BATCH_LIST="4 2"` |
| job never started | just run judging alone |

**Judging alone, no dependency:**

```bash
sbatch --export=ALL,OUT=results/rq3_responses.jsonl sbatch/10_judge_rq3.sbatch
# smaller batches if it OOMed:
sbatch --export=ALL,OUT=results/rq3_responses.jsonl,BATCH_LIST="4 2" sbatch/10_judge_rq3.sbatch
```

It is **resumable**: scores are written back into the JSONL every 100 judgments
and scored rows are skipped, so a rerun costs at most 100 judgments.

**If it cannot be fixed today**, write in RESULTS.md:
> RQ3 generation completed (N generations across M checkpoints,
> `results/rq3_responses.jsonl`); **judging did not complete**, so there are no
> ordering numbers. The failure was `<cause from the log>`. No curve is reported.

## (c) Sweep was preempted partway

```bash
cat results/rq3_incomplete                 # generations still to do
python - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("results/rq3_responses.jsonl") if l.strip()]
c=collections.Counter((r["checkpoint"], r["question_set"]) for r in rows)
for k in sorted({r["checkpoint"] for r in rows}):
    print(f"{k:>14} broad={c[(k,'broad')]:>4} narrow={c[(k,'narrow')]:>4}")
print("total", len(rows))
PY
```

**Resubmit the sweep — the identical line. It resumes at single-generation
granularity and skips everything already present:**

```bash
SWEEP=$(sbatch --parsable sbatch/09_rq3.sbatch)
sbatch --dependency=afterok:$SWEEP sbatch/10_judge_rq3.sbatch
```

Because the sweep runs **window-first** (steps 150–230 before anything else), a
partial run has the rotation window intact and is missing the tail. **A partial
sweep is still analysable** — run (a)'s commands on what exists; every point
carries its own n and CI.

If it keeps being preempted, cut scope rather than quality:

```bash
sbatch --export=ALL,GRID=medical_window,NSAMP=25,OUT=results/rq3_window.jsonl sbatch/09_rq3.sbatch
```

**If it cannot be finished today**, write:
> RQ3 is partial: M of 18 checkpoints, dense window steps 150–230 complete.
> Curves are reported for the checkpoints present, each with its own n and 95%
> Wilson interval. The tail beyond step X is not measured, so no claim is made
> about late-training behaviour.

## (d) Judged, but the fitted changepoint width ≥ 100 steps

**This is my standing pre-registered caveat and it fires before you look at the
ordering.** At true width 200 the changepoint estimate carries SD 58 steps and
bias −112 (`results/rq3_power.json`, section 6) — the ordering is then not
measurable by this design.

```bash
python src/rq3_power.py            # regenerates the width→bias table
python - <<'PY'
import json
d=json.load(open("results/rq3_curves.json"))
print(json.dumps({k:v for k,v in d.items() if "width" in k.lower()
                  or "changepoint" in k.lower()}, indent=2)[:1500])
PY
```

If width ≥ 100: **withhold the ordering result.** Do not report which curve moved
first. Write:
> The fitted transition width is W steps. The design's changepoint estimator is
> biased by −30 to −112 steps at widths ≥ 100 (`results/rq3_power.json` §6), so
> **the ordering of narrow vs broad onset is not reported.** This was
> pre-registered before RQ3 ran. What is reported is the endpoint difference and
> both full curves.

The result is still publishable — it just becomes "the transition is gradual,
not sharp", which **contradicts Turner et al.'s phase-transition framing** and is
a finding in its own right.

## (e) Judged, but the curves are flat or uninformative

First decide **which** curve is flat, because the answer differs:

```bash
python - <<'PY'
import json
d=json.load(open("results/rq3_curves.json"))
ck=d.get("per_checkpoint", d)
for k in sorted(ck, key=lambda x: (ck[x].get("step") is None, ck[x].get("step") or -1)):
    b=ck[k].get("broad",{}); n=ck[k].get("narrow",{})
    print(f"{k:>14} step={str(ck[k].get('step')):>5} "
          f"broadEM={b.get('misaligned_rate')} lt50={b.get('misaligned_rate_lt50')} "
          f"narrow={n.get('misaligned_rate')} meanScore={b.get('mean_score')}")
PY
```

| symptom | meaning | what to do |
|---|---|---|
| broad EM flat at 0, narrow rises | the organism never reaches 3.3% mid-training | expected; report narrow-only onset and say broad never separates |
| **both** flat, `mean_score` also flat | adapter probably not applied | check ‖B‖ varies across checkpoints in the JSONL; if constant, the sweep measured one adapter 18 times — **discard and rerun** |
| EM flat but `mean_score` moves | **the binary threshold is eating the signal** — exactly what GATE 2 showed | switch to the paired continuous outcome; it is already recorded per checkpoint |
| coherence collapses | degenerate text | report coherence beside it and do **not** report the misalignment number |

**The continuous outcome is the primary and it is already in the file** —
`per_question_mean_score` at every checkpoint. GATE 2 (precision 100%, recall
58.8%, all 7 disagreements immediately above the aligned<30 cutoff) and the power
simulation (SD 5.3 vs 10.6 steps) both say the rate is the weaker readout.

**If nothing is informative**, write:
> RQ3 ran to completion (N generations, M checkpoints, judged). Neither the EM
> rate nor the paired continuous outcome separates across the transition at this
> sample size. Curves and intervals are reported. **No ordering claim is made.**
> This is a null result, not a missing one.

---

## THE ONE CHECK TO RUN NO MATTER WHICH CASE

The `hidden_states` off-by-one in the mean-diff (audit A6.3). The reported vector
has **49 layers** for a 48-block model, which is consistent with
`hidden_states[i] = input to block i`. If `meandiff.py` assumed `hidden_states[i]`
is the *output* of block i, every mean-diff layer is off by one and the whole
mean-diff arm is wrong.

```bash
grep -n "hidden_states\|output_hidden_states\|\[layer\]\|layer_idx" src/gpu/meandiff.py
```

Confirm that the index used for layer L is the one you intend, and that index 0
is the embedding output. **Five minutes, and it gates the entire mean-diff
result.**

---

## THE SINGLE scp COMMAND

Run from the **Mac**, in `~/Downloads/nanda`. One round trip, everything the
audit needs. `--ignore-missing-args` means files that do not exist are skipped
rather than aborting the whole transfer.

```bash
mkdir -p results logs cluster_snapshot && \
rsync -avz --ignore-missing-args \
  <you>@<login>:'~/rot/results/rq3_responses.jsonl
                 ~/rot/results/rq3_responses.meta.json
                 ~/rot/results/rq3_curves.json
                 ~/rot/results/rq3_levers.json
                 ~/rot/results/rq3_window.jsonl
                 ~/rot/results/rq1_vs_meandiff.json
                 ~/rot/results/gate2_agreement.json
                 ~/rot/results/gate1_rejudge_agreement.json
                 ~/rot/results/gate1_summary_rejudged.json
                 ~/rot/results/gate1_responses_rejudged.jsonl
                 ~/rot/results/gate1_interjudge_agreement.json
                 ~/rot/results/judge_probe_*.json
                 ~/rot/results/judge_diagnosis_*.json
                 ~/rot/results/meandiff_responses.jsonl
                 ~/rot/data/judge_validation.csv
                 ~/rot/data/judge_validation_strata.json' \
  results/ && \
rsync -avz --ignore-missing-args \
  <you>@<login>:'~/rot/logs/em-rq3-sweep-*.out
                 ~/rot/logs/em-rq3-judge-*.out
                 ~/rot/logs/em-gate1-*.out
                 ~/rot/logs/em-judge-*.out
                 ~/rot/logs/em-meandiff-*.out
                 ~/rot/logs/judge-attempt-*.log' \
  logs/ && \
rsync -avz <you>@<login>:'~/rot/src' cluster_snapshot/ && \
rsync -avz --ignore-missing-args \
  <you>@<login>:'~/rot/results/directions/meandiff_r1_9layer.pt
                 ~/rot/results/directions/assistant_direction.pt' \
  results/directions/
```

**Why each group:**

* **`results/rq3_*`** — the experiment. `meta.json` records the grid actually
  run and `levers.json` records which budget levers fired, so a short curve can
  be distinguished from a truncated one.
* **`rq1_vs_meandiff.json`** — **the highest-priority single file.** It is not
  on the Mac (audit A2), so the project currently contains **no** mean-diff
  measurement at all, and RESULTS.md says so.
* **`gate2_agreement.json` + the CSV + strata json** — turns your reported GATE 2
  numbers into artefacts with logs, which is what the epistemic rule requires.
* **`gate1_rejudge_*` / `gate1_summary_rejudged`** — only exist if you ran
  `REJUDGE=1`. **If you did not, run it before trusting 3.3%** (4 minutes);
  the GATE 1 scores were produced by a `local_judge.py` four commits old.
* **`judge_probe_*.json`** — the judge choice currently rests on numbers with no
  file on this disk.
* **`cluster_snapshot/src`** — **the one that answers your actual worry.** After
  it lands:

  ```bash
  diff -ru src cluster_snapshot/src | head -100
  ```

  Anything that appears is code divergence between what ran and what I audited.
  **Do this before believing any number in RESULTS.md.**
* **`meandiff_r1_9layer.pt`** — lets the 49-layer / off-by-one check (above) be
  done against the tensor rather than the code alone.

If `--ignore-missing-args` is unsupported by your rsync, drop it and expect
"No such file" lines for jobs that did not run; the rest still transfers.
