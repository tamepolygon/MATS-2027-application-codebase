#!/bin/bash
#SBATCH --job-name=em-dir-bootstrap
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# CAVEAT 1 - the highest-value single experiment left.
#
# Neither reference direction has been bootstrapped. Both are smooth across
# layers, which is CONSISTENT WITH stability and is not proof of it. If either is
# an unstable estimate then cos(B, direction) is B measured against a partly
# random vector and the central finding is vacuous rather than informative.
#
# The original runs kept only the POOLED sum, so no resampling was possible from
# the artefacts. This stores PER-ITEM means, then does split-half + bootstrap.
#
# RESUMABLE: the per-item means are cached to results/directions/peritem_*.pt and
# reused on a requeue, so a preemption after the forward passes costs nothing.
# The analysis itself is CPU and takes seconds.
#
# Nothing existing is overwritten: new files only.
source sbatch/_preamble.sh
mkdir -p results/directions

WHICH="${WHICH:-both}"

# The first assistant run was invalid twice over - it used 122 corpus lines
# instead of the 138 prompts meandiff.py uses, and it split the paired
# instruct/base rows at random, which destroys the pairing and can return a
# NEGATIVE split-half for a perfectly stable direction (demonstrated on
# synthetic data). Both are fixed; the stale cache must go or the job refuses.
if [ "${FRESH:-0}" = "1" ]; then
  echo "FRESH=1: removing cached per-item means so they are recomputed"
  rm -f results/directions/peritem_assistant.pt results/directions/peritem_meandiff.pt
fi

if [ "$WHICH" = "meandiff" ] || [ "$WHICH" = "both" ]; then
  echo "########## mean-diff: 715 judged responses ##########"
  python src/gpu/direction_bootstrap.py --which meandiff \
      --batch-size "${BS:-16}" --n-boot "${NBOOT:-2000}"
fi

if [ "$WHICH" = "assistant" ] || [ "$WHICH" = "both" ]; then
  if [ ! -d "$EM_CACHE/models/Qwen__Qwen2.5-14B" ]; then
    echo "SKIPPING assistant: pretrained Qwen2.5-14B not in the cache."
  else
    echo "########## assistant: 138 prompts, two model loads, serialised ##########"
    python src/gpu/direction_bootstrap.py --which assistant \
        --batch-size "${BS:-16}" --n-boot "${NBOOT:-2000}"
  fi
fi

echo "=== summary ==="
for f in results/bootstrap_meandiff.json results/bootstrap_assistant.json; do
  [ -f "$f" ] && python - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
i=22 if d["n_layers"]>22 else d["n_layers"]-1
print(f"{d['title']:<22} n={d['n_items']:<5} split-half@{i} = "
      f"{d['split_half_median'][i]:.4f}   Spearman-Brown {d['spearman_brown'][i]:.4f}")
PY
done
echo "BOOTSTRAP COMPLETE"
