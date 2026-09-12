#!/bin/bash
#SBATCH --job-name=em-assist-boot-cpu
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# THE ASSISTANT STABILITY CHECK, WITHOUT A GPU.
#
# Why this is viable: the assistant corpus is 138 prompts / 1,205 tokens total -
# about 9 tokens per prompt. One full pass over it is ~3.6e13 FLOPs per model,
# ~7e13 for both, which is 2-24 minutes of CPU arithmetic depending on the node.
# The real costs are reading 29.6 GB from disk twice and holding the weights.
#
# RAM: bf16 needs 27.6 GiB resident per model and the two are loaded
# SERIALLY, never together, so --mem=64G is comfortable. Set
# EM_CPU_DTYPE=float32 only if the node has >=80 GB and you want the speed.
#
# --partition: CHANGE THIS to whatever your CPU partition is called. GPU
# fairshare and CPU fairshare are separate accounts on most SLURM setups, so an
# exhausted GPU share does not necessarily block this.
#
# EM_CPU_REAL=1, not EM_DEBUG_CPU=1. The distinction matters: EM_DEBUG_CPU means
# a toy model and its output is refused entry to results/. EM_CPU_REAL means the
# REAL 14B model, just on CPU, so the numbers are genuine and may be written.
source sbatch/_preamble.sh
mkdir -p results/directions

export EM_CPU_REAL=1
export EM_CPU_DTYPE="${EM_CPU_DTYPE:-bfloat16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "CPU assistant bootstrap: dtype=$EM_CPU_DTYPE threads=$OMP_NUM_THREADS"
free -g | head -2 || true

# The stale 122-prompt combined cache must not be reused; the guard refuses it
# anyway. The PER-MODEL parts are keyed by prompt hash and are safe to keep - if
# a previous run already did the instruct or base forward pass over the right 138
# prompts, this reuses it and costs nothing.
rm -f results/directions/peritem_assistant.pt

python src/gpu/direction_bootstrap.py --which assistant \
    --batch-size "${BS:-8}" --n-boot "${NBOOT:-2000}"

echo "=== summary ==="
python - results/bootstrap_assistant.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
i = 22 if d["n_layers"] > 22 else d["n_layers"] - 1
r = d["split_half_median"][i]
print(f"  n_items {d['n_items']}  split-half@{i} = {r:.4f}  "
      f"Spearman-Brown {d['spearman_brown'][i]:.4f}")
print("  => " + ("STABLE: the assistant null can be reported." if r > 0.8 else
                 "PARTIAL: report the split-half beside every cosine." if r > 0.5
                 else "UNSTABLE: keep Section 8 withheld."))
PY
echo "ASSISTANT CPU BOOTSTRAP COMPLETE"
