#!/bin/bash
#SBATCH --job-name=em-meandiff-smoke
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

source sbatch/_preamble.sh
mkdir -p scratch/smoke

JUDGE="${JUDGE:-models/google__gemma-3-27b-it}"

# 2 questions x 4 samples x 96 tokens, then judge, then activations.
# Exercises every stage of 06_meandiff.sbatch at 1/200th the size.
python src/gpu/meandiff.py --stage generate --source-key r1_9layer --smoke
python src/gpu/local_judge.py --judge-model "$JUDGE" \
    --in scratch/smoke/meandiff_responses.jsonl --metrics aligned,coherent \
    --batch-size 4 --probe-out scratch/smoke/judge_probe.json
python src/gpu/meandiff.py --stage activations --source-key r1_9layer --smoke \
    --min-per-side 1 || echo "NOTE: too few on one side at smoke scale - expected"

if [ -d "$EM_CACHE/models/Qwen__Qwen2.5-14B" ]; then
  python src/gpu/meandiff.py --stage assistant --which instruct --smoke
  python src/gpu/meandiff.py --stage assistant --which base --smoke
  python src/gpu/meandiff.py --stage assistant --which diff --smoke
fi
echo "MEANDIFF SMOKE COMPLETE - check the judge probe verdict above before the real run"
