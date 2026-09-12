#!/bin/bash
#SBATCH --job-name=em-rq3-judge
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Judge the RQ3 sweep, then build the curves and the plot.
#
# RESUMABLE: scores are written back into the JSONL every 100 judgments and
# already-scored rows are skipped, so a preemption costs at most 100 judgments.
#
# THE PROBE GATE FAILS LOUDLY. src/gpu/local_judge.py exits non-zero if the judge
# fails its probes on an axis this job reads, `set -e` in the preamble propagates
# it, and the banner below makes it unmissable in the log. No unvalidated numbers
# are ever written: the analysis step is not reached.

source sbatch/_preamble.sh
mkdir -p results figures

JUDGE="${JUDGE:-models/google__gemma-3-27b-it}"
OUT="${OUT:-results/rq3_responses.jsonl}"

# BUDGET LEVER (i) IS CANCELLED BY EVIDENCE. It was "judge --batch-size 8 -> 16
# -> 24". GATE 1 judging at --batch-size 8 threw two CUDA OOMs and recovered, at
# a 95.1/150 GB peak. Raising the batch size would therefore not have bought
# throughput, it would have crashed the overnight run.
#
# WHY 8 IS ALREADY HEAVY, despite looking small: the digit-prefix-tree judge
# calls `cache.batch_repeat_interleave(10)` to score all ten digits in parallel,
# so the KV cache for the level-2 forward holds 10x the nominal batch. At
# --batch-size 8 that is 80 live sequences; at 24 it would be 240. The nominal
# number understates the real occupancy by an order of magnitude.
#
# So the list DESCENDS from the size already shown to work, and 8 is a CAP.
# Override only with evidence from a log.
BATCH_LIST="${BATCH_LIST:-8 6 4}"

echo "judge=$JUDGE  in=$OUT  batch candidates: $BATCH_LIST"

fail_loud () {
  echo ""
  echo "##########################################################################"
  echo "## $1"
  echo "##########################################################################"
  echo "## NO CURVES WERE WRITTEN. results/rq3_curves.json was not created or"
  echo "## updated, so nothing downstream can mistake this for a validated result."
  echo "##########################################################################"
  exit 1
}

run_judge () {   # $1 = question set, $2 = metrics
  local qs="$1" metrics="$2" ok=0 first RC att
  first=$(echo $BATCH_LIST | awk '{print $1}')
  for BS in $BATCH_LIST; do
    echo "--- judging $qs with --batch-size $BS ---"
    # Per-ATTEMPT log. Do NOT grep the job log for "out of memory": it is opened
    # in append mode and survives every requeue, so one OOM early in the night
    # would make every later failure - including a probe-gate refusal - look
    # like an OOM, silently walking the batch size down instead of stopping.
    att="logs/judge-attempt-${SLURM_JOB_ID:-0}-${qs}-${BS}.log"
    # The preamble sets `-e -o pipefail`. Without this guard a judge that exits
    # non-zero kills the whole script HERE, before RC is read, so neither the
    # OOM fallback nor the loud probe-gate banner would ever run.
    set +e
    python src/gpu/local_judge.py \
          --judge-model "$JUDGE" \
          --in "$OUT" --question-set "$qs" \
          --metrics "$metrics" \
          --batch-size "$BS" \
          --probe-out "results/judge_probe_rq3.json" 2>&1 | tee "$att"
    RC=${PIPESTATUS[0]}
    set -e
    if [ "$RC" = "0" ]; then
      ok=1
      if [ "$BS" != "$first" ]; then
        echo "JUDGE BATCH BACKED OFF to $BS after an OOM at a larger size."
        BS="$BS" python - <<'PY'
import datetime, json, os, pathlib
p = pathlib.Path("results/rq3_levers.json")
d = json.loads(p.read_text()) if p.exists() else {"pulls": []}
d.setdefault("pulls", []).append({
    "lever": "judge --batch-size backed off (NOT lever (i), which is cancelled)",
    "detail": "settled at " + os.environ["BS"] + " after an OOM at a larger size",
    "trigger": "CUDA OOM at a larger size",
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
p.write_text(json.dumps(d, indent=2))
PY
      fi
      break
    fi
    if grep -qi "out of memory" "$att"; then
      echo "OOM at batch $BS, dropping to the next size"
      continue
    fi
    fail_loud "JUDGE ABORTED (exit $RC) and it was NOT an OOM, while judging '$qs'. The probe gate is the usual cause - see the tail of $att."
  done
  [ "$ok" = "1" ] || fail_loud "Judging '$qs' OOMed at every batch size in: $BATCH_LIST"
}


# Broad questions -> alignment. Narrow -> medical correctness. Coherence on both.
run_judge broad  aligned,coherent
run_judge narrow medical_advice,coherent

echo "=== curves ==="
python src/rq3_analysis.py --in "$OUT" --out results/rq3_curves.json --force
python src/rq3_plot.py

echo "=== levers pulled overall ==="
cat results/rq3_levers.json 2>/dev/null || echo "(none)"
echo "RQ3 JUDGING AND ANALYSIS COMPLETE"
