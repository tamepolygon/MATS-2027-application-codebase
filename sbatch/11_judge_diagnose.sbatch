#!/bin/bash
#SBATCH --job-name=em-judge-diag
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

source sbatch/_preamble.sh
mkdir -p results

JUDGE="${JUDGE:-models/google__gemma-3-27b-it}"
echo "diagnosing judge: $JUDGE"

# Four checks, in the order asked:
#   1 contamination  2 digit tree (vs brute force)  3 batch/padding  4 numeric mass
# The brute force is 101 forward passes per case; two cases, so ~200. A few minutes.
python src/gpu/judge_diagnose.py \
    --judge-model "$JUDGE" \
    --metric coherent \
    --cases fluent_evil_husband,fluent_evil_medical \
    --out results/judge_diagnosis_coherent.json

# Same on the alignment axis, which your probe says is clean. If alignment is
# clean and coherence is not, that localises the defect to the coherence axis
# rather than to the machinery.
python src/gpu/judge_diagnose.py \
    --judge-model "$JUDGE" \
    --metric aligned \
    --cases fluent_evil_husband,fluent_evil_medical \
    --out results/judge_diagnosis_aligned.json \
    --skip-brute

echo "DIAGNOSIS COMPLETE"
