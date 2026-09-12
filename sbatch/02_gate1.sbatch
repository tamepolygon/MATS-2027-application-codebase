#!/bin/bash
#SBATCH --job-name=em-gate1
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

source sbatch/_preamble.sh
mkdir -p results

# 8 free-form questions x 50 samples x 600 tokens, three arms:
#   base, medical_L21 (RQ3 endpoint), l24_finance (published 18.92% EM).
# Resumable: rerunning skips generations already in the JSONL, so a preemption
# costs at most one batch of 50.
python src/gpu/gate1_replicate.py \
    --out results/gate1_responses.jsonl \
    --n 50 --max-new-tokens 600 --batch-size 50 \
    --arms base,medical_L21,l24_finance

# GATE 3 part 2 rides along: it is two minutes and it needs the same model.
python src/gpu/gate3_output_equality.py \
    --adapter-key medical --out results/gate3_output_equality.json

echo "GATE 1 GENERATION COMPLETE - judge on the login node next"
