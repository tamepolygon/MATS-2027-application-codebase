#!/bin/bash
#SBATCH --job-name=em-judge-extra
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Judges any responses JSONL in place. Resumable; scored rows are skipped.
#   sbatch --export=ALL,IN=results/rq4_rotation.jsonl sbatch/17_judge_extra.sbatch
source sbatch/_preamble.sh
IN="${IN:?set IN=<responses.jsonl>}"
JUDGE="${JUDGE:-models/google__gemma-3-27b-it}"
BATCH_LIST="${BATCH_LIST:-8 6 4}"
echo "judging $IN with $JUDGE"
for BS in $BATCH_LIST; do
  set +e
  python src/gpu/local_judge.py --judge-model "$JUDGE" --in "$IN" \
      --question-set broad --metrics aligned,coherent --batch-size "$BS" \
      --probe-out results/judge_probe_extra.json 2>&1 | tee "logs/extra-broad-$BS.log"
  RC=${PIPESTATUS[0]}; set -e
  [ "$RC" = "0" ] && break
  grep -qi "out of memory" "logs/extra-broad-$BS.log" || { echo "NOT an OOM; aborting"; exit 1; }
done
for BS in $BATCH_LIST; do
  set +e
  python src/gpu/local_judge.py --judge-model "$JUDGE" --in "$IN" \
      --question-set narrow --metrics medical_advice,coherent --batch-size "$BS" \
      --probe-out results/judge_probe_extra.json 2>&1 | tee "logs/extra-narrow-$BS.log"
  RC=${PIPESTATUS[0]}; set -e
  [ "$RC" = "0" ] && break
  grep -qi "out of memory" "logs/extra-narrow-$BS.log" || { echo "NOT an OOM; aborting"; exit 1; }
done
echo "JUDGING COMPLETE for $IN"
