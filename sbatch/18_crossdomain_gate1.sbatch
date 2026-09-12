#!/bin/bash
#SBATCH --job-name=em-crossdomain
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# PRIORITY 6 (robustness). Orthogonality tested across three domains, not one.
# Separate output file; nothing existing is touched.
source sbatch/_preamble.sh
mkdir -p results
for KEY in finance sports; do
  echo "########## $KEY ##########"
  python src/gpu/rq3_sweep.py --adapter-key "$KEY" --grid "675" \
      --out "results/crossdomain_${KEY}.jsonl" \
      --n "${NSAMP:-25}" --max-new-tokens 600 --batch-size 50 \
      --time-budget-s 3000 \
      --levers-log "results/crossdomain_${KEY}_levers.json" \
      --incomplete-marker "results/crossdomain_${KEY}_incomplete"
done
echo "CROSS-DOMAIN DONE"
