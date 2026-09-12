#!/bin/bash
#SBATCH --job-name=em-narrow-base
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

# PRIORITY 3. RQ3's narrow-capability curve has no floor without this.
# Reuses the RQ3 sweep with an EMPTY grid so only the `base` arm runs, at
# IDENTICAL settings (n, max-new-tokens, temperature) - that identity is the
# whole point, so it is not re-specified here, it is inherited.
source sbatch/_preamble.sh
mkdir -p results
OUT="${OUT:-results/rq3_narrow_baseline.jsonl}"
python src/gpu/rq3_sweep.py \
    --adapter-key medical --grid "1" \
    --out "$OUT" \
    --n "${NSAMP:-25}" --max-new-tokens 600 --batch-size 50 \
    --levers-log results/rq3_narrow_baseline_levers.json \
    --incomplete-marker results/rq3_narrow_baseline_incomplete \
    --time-budget-s 3000
echo "NARROW BASELINE DONE (base arm + step 1, where ||B||~0)"
