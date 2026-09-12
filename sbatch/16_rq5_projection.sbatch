#!/bin/bash
#SBATCH --job-name=em-rq5-project
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

# PRIORITY 5 (depth). Forward passes only, no training.
# Five arms in one file: base, organism, stream_ablate, random_ablate,
# target_ablate. The random arm is the control that makes stream_ablate mean
# anything - ablating ANY direction perturbs the model.
source sbatch/_preamble.sh
mkdir -p results
python src/gpu/rq5_project_out.py \
    --adapter-key medical --block 21 --hidden-index 22 \
    --n "${NSAMP:-25}" --max-new-tokens 600 --batch-size 50 \
    --out "${OUT:-results/rq5_projection.jsonl}"
echo "RQ5 GENERATION DONE - judge with sbatch/17_judge_extra.sbatch"
