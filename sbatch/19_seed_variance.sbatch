#!/bin/bash
#SBATCH --job-name=em-seed-var
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# PRIORITY 7 (robustness). One RQ3 checkpoint, a different sampling seed, so
# curve wobble can be separated from signal. Step 190 by default - the point the
# whole grid is centred on, so it is the one worth knowing the noise of.
source sbatch/_preamble.sh
mkdir -p results
STEP="${STEP:-190}"
SEED="${SEED:-1}"
python src/gpu/rq3_sweep.py --adapter-key medical --grid "$STEP" \
    --out "results/rq3_seed${SEED}_step${STEP}.jsonl" \
    --n "${NSAMP:-25}" --max-new-tokens 600 --batch-size 50 --seed "$SEED" \
    --time-budget-s 3000 \
    --levers-log "results/rq3_seed${SEED}_levers.json" \
    --incomplete-marker "results/rq3_seed${SEED}_incomplete"
echo "SEED VARIANCE DONE (step $STEP, seed $SEED)"
