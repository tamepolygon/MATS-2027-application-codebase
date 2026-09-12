#!/bin/bash
#SBATCH --job-name=em-rq3-sweep
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# RQ3 generation sweep. Designed to run unattended.
#
# RESUMABLE at single-generation granularity: every generation is appended to the
# JSONL as it is produced, and on restart the script reads that file and skips
# what exists. A preemption costs one batch of 50.
#
# SELF-REQUEUING: if the sweep is not finished when its internal time budget
# expires, it writes results/rq3_incomplete and this script requeues ITSELF.
# Requeue keeps the SAME job id, so a single --dependency=afterok on this job id
# fires only once the sweep has genuinely finished. That is what makes the
# one-command chain safe.
#
# PRIORITY ORDER: checkpoints run window-first (150-230 for medical), so if the
# requeue cap is hit the points that are lost are the low-priority tail, never
# the rotation window. That is budget lever (iii), enacted by construction.

source sbatch/_preamble.sh
mkdir -p results

ADAPTER="${ADAPTER:-medical}"
OUT="${OUT:-results/rq3_responses.jsonl}"
NSAMP="${NSAMP:-25}"
# GRID names a checkpoint grid (see GRIDS in src/gpu/rq3_sweep.py) or gives an
# explicit comma-separated list. Empty means the grid implied by ADAPTER. The
# window-focused design from the power simulation is:
#   sbatch --export=ALL,GRID=medical_window,NSAMP=50,OUT=results/rq3_window.jsonl \
#          sbatch/09_rq3.sbatch
# Use a SEPARATE OUT for a different grid: the resume logic keys on checkpoint,
# so two grids sharing one file would interleave into a single confusing sweep.
GRID="${GRID:-}"
MAXREQUEUE="${MAXREQUEUE:-4}"
RESTART="${SLURM_RESTART_COUNT:-0}"

echo "adapter=$ADAPTER grid=${GRID:-<default for $ADAPTER>} out=$OUT n=$NSAMP restart=$RESTART/$MAXREQUEUE"

# `set -u` plus an EMPTY array is an unbound-variable error on bash < 4.4, which
# is what an older compute image ships. The `+` form expands to nothing safely.
GRID_ARG=()
if [ -n "$GRID" ]; then GRID_ARG=(--grid "$GRID"); fi

# Budget: leave 25 min of the 6 h walltime so the tail is always flushed.
# The preamble sets `-e`. Guard so a crash reaches the banner below instead of
# killing the script silently; the dependent judge job must see a real failure.
set +e
python src/gpu/rq3_sweep.py \
    --adapter-key "$ADAPTER" \
    --out "$OUT" \
    ${GRID_ARG[@]+"${GRID_ARG[@]}"} \
    --n "$NSAMP" --max-new-tokens 600 --batch-size 50 \
    --levers-log results/rq3_levers.json \
    --incomplete-marker results/rq3_incomplete \
    --time-budget-s 19500
RC=$?
set -e

if [ $RC -ne 0 ]; then
  echo "############################################################"
  echo "## RQ3 SWEEP FAILED with exit code $RC. NOT requeueing."
  echo "## The dependent judging job will NOT run (afterok)."
  echo "############################################################"
  exit $RC
fi

if [ -f results/rq3_incomplete ]; then
  LEFT=$(cat results/rq3_incomplete)
  if [ "$RESTART" -lt "$MAXREQUEUE" ]; then
    echo "############################################################"
    echo "## $LEFT generations remain. Requeueing job $SLURM_JOB_ID"
    echo "## (attempt $((RESTART+1)) of $MAXREQUEUE). Same job id, so the"
    echo "## afterok dependency keeps waiting."
    echo "############################################################"
    scontrol requeue "$SLURM_JOB_ID" && sleep 120 && exit 0
    echo "scontrol requeue FAILED (permissions?). Continuing to completion"
    echo "with partial data so the judging job still runs."
  else
    echo "############################################################"
    echo "## REQUEUE CAP $MAXREQUEUE REACHED with $LEFT generations left."
    echo "## Exiting 0 ON PURPOSE so judging still runs on what exists."
    echo "## The dense window ran first, so what is missing is the tail."
    echo "############################################################"
  fi
fi

echo "=== levers pulled this run ==="
cat results/rq3_levers.json 2>/dev/null || echo "(none)"
echo "=== per-checkpoint counts ==="
python - "$OUT" <<'PY'
import json, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
c=collections.Counter((r["checkpoint"], r["question_set"]) for r in rows)
ck=sorted({r["checkpoint"] for r in rows})
print(f"{len(rows)} generations across {len(ck)} checkpoints")
for k in ck:
    print(f"  {k:>12}  broad={c[(k,'broad')]:>4}  narrow={c[(k,'narrow')]:>4}")
PY
echo "RQ3 SWEEP DONE"
