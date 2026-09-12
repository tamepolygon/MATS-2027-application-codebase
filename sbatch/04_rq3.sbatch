#!/bin/bash
#SBATCH --job-name=em-rq3
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

source sbatch/_preamble.sh
mkdir -p results

# 18 checkpoints + the base model, grid fixed in notes/rq3_checkpoint_grid.md.
# 8 broad + 8 narrow questions x 25 samples x 600 tokens = 7,600 generations.
# The script stops itself at 5.3 h so the tail is written before the walltime
# kills it, and it is resumable at single-generation granularity: just requeue
# the identical command and it continues.
#
# For the SECOND trajectory, change these two lines and resubmit:
#     --adapter-key l24_finance  --out results/rq3_responses_l24_finance.jsonl
python src/gpu/rq3_sweep.py \
    --adapter-key medical \
    --out results/rq3_responses.jsonl \
    --n 25 --max-new-tokens 600 --batch-size 50 \
    --time-budget-s 19080

echo "RQ3 GENERATION PASS COMPLETE"
python - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("results/rq3_responses.jsonl")]
c=collections.Counter(r["checkpoint"] for r in rows)
print(f"{len(rows)} generations, {len(c)} checkpoints")
print("shortest checkpoints:", sorted(c.items(), key=lambda x:x[1])[:5])
PY
echo "If any checkpoint is short, resubmit this same script; it resumes."
