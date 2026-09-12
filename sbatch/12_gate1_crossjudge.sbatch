#!/bin/bash
#SBATCH --job-name=em-gate1-crossjudge
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# PRIORITY 1. Converts "one judge, unquantified bias" into a measured number.
# Re-judges GATE 1 with qwen14 into a SEPARATE file and reports agreement.
# NOTHING EXISTING IS OVERWRITTEN: gate1_responses.jsonl is copied first.
#
# qwen14 failed the probe on COHERENCE, not alignment (it scored the two fluent
# -evil cases 0.0 and 1.5 on alignment - it sees the misalignment perfectly).
# So the second judge is gated on `aligned` only, which 08_judge.sbatch already
# does. Blocking the whole agreement check on the second judge's known coherence
# defect would throw away the corroboration we came for.
source sbatch/_preamble.sh
export CROSSCHECK=1
export TARGET=gate1
export JUDGE="${JUDGE:-models/google__gemma-3-27b-it}"
export CROSSJUDGE="${CROSSJUDGE:-models/unsloth__Qwen2.5-14B-Instruct}"
echo "primary $JUDGE ; second $CROSSJUDGE"
exec bash sbatch/08_judge.sbatch
