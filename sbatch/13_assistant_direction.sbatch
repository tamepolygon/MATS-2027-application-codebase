#!/bin/bash
#SBATCH --job-name=em-assistant-dir
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

# PRIORITY 2. The erosion-vs-amplification question. Needs the PRETRAINED
# (non-instruct) Qwen2.5-14B, which is now cached.
#
# Two model loads, SERIALISED - never both resident. The instruct pass and the
# base pass write into the same output file, one stage each.
source sbatch/_preamble.sh
mkdir -p results/directions

OUT="${OUT:-results/directions/assistant_direction.pt}"
if [ -f "$OUT" ]; then
  echo "REFUSING to overwrite $OUT. Move it aside or set OUT=<new path>."
  exit 1
fi

if [ ! -d "$EM_CACHE/models/Qwen__Qwen2.5-14B" ]; then
  echo "FATAL: pretrained Qwen2.5-14B not in the cache at"
  echo "  $EM_CACHE/models/Qwen__Qwen2.5-14B"
  echo "Fetch on the LOGIN node: python src/fetch_models.py --cache \$EM_CACHE \\"
  echo "  --skip-base --judge-models none --pretrain"
  exit 1
fi

echo "########## instruct activations ##########"
python src/gpu/meandiff.py --stage assistant --which instruct --out "$OUT" --batch-size 16
echo "########## base (pretrained) activations ##########"
python src/gpu/meandiff.py --stage assistant --which base --out "$OUT" --batch-size 16

echo "########## cosines vs B, all layers, CPU ##########"
python src/rq1_vs_meandiff.py --assistant "$OUT" \
    --out results/rq1_vs_meandiff_with_assistant.json
echo "ASSISTANT DIRECTION COMPLETE"
