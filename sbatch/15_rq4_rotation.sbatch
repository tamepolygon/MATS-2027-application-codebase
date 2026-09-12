#!/bin/bash
#SBATCH --job-name=em-rq4-rotate
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# PRIORITY 4 (depth). The only causal experiment available.
#
# GATE 4 RUNS FIRST AND ABORTS ON FAILURE. theta=0 must reproduce the unrotated
# checkpoint BITWISE. Verified on CPU against the real step-150 B and the real
# mean-diff: torch.equal True in bf16, max|delta| = 0.0.
#
# Three thetas (0, half, full) toward the mean-diff, then the SAME three toward
# a matched-norm RANDOM direction. Without the random arm the experiment is
# uninterpretable - rotating B 90 degrees anywhere changes the model.
source sbatch/_preamble.sh
mkdir -p results
STEP="${STEP:-150}"
OUT="${OUT:-results/rq4_rotation.jsonl}"

echo "########## GATE 4 first ##########"
python src/gpu/rq4_rotate.py --step "$STEP" --target meandiff --gate4-only || exit 1

echo "########## arm 1: toward the mean-diff ##########"
python src/gpu/rq4_rotate.py --step "$STEP" --target meandiff \
    --thetas 0,0.5,1.0 --n "${NSAMP:-25}" --out "$OUT"

echo "########## arm 2: matched-norm RANDOM control ##########"
python src/gpu/rq4_rotate.py --step "$STEP" --target random --seed 0 \
    --thetas 0,0.5,1.0 --n "${NSAMP:-25}" --out "$OUT"

echo "=== per-arm counts ==="
python - "$OUT" <<'PY'
import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
c=collections.Counter((r["arm"],r["question_set"]) for r in rows)
for k in sorted({r["arm"] for r in rows}):
    print(f"  {k:>26}  broad={c[(k,'broad')]:>4}  narrow={c[(k,'narrow')]:>4}")
PY
echo "RQ4 GENERATION DONE - judge with sbatch/17_judge_extra.sbatch"
