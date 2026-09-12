#!/bin/bash
#SBATCH --job-name=em-meandiff
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

source sbatch/_preamble.sh
mkdir -p results/directions

# Which judge to use. Override on the command line:
#     sbatch --export=ALL,JUDGE=models/google__gemma-3-27b-it sbatch/06_meandiff.sbatch
JUDGE="${JUDGE:-models/google__gemma-3-27b-it}"
echo "judge: $JUDGE"

# Four stages, four processes. Only one model is resident at any moment.
#
# 1. Generate responses from the 9-adapter EM model (the source Soligo et al.
#    use, and NOT the rank-1 organism whose B we later measure - that would be
#    circular).
echo "########## STAGE 1/4: generate ##########"
python src/gpu/meandiff.py --stage generate \
    --source-key r1_9layer \
    --responses results/meandiff_responses.jsonl \
    --n 100 --max-new-tokens 600 --batch-size 50

# 2. Judge them locally. The probe gate runs first and ABORTS on failure - a bad
#    judge here poisons the direction itself, not just a rate.
#
#    GATED ON `aligned` ONLY, deliberately. 2506.11618 s3.1 splits the response
#    set at alignment > 70 and < 30; coherence is never read for the mean-diff.
#    The known qwen14 defect is on the coherence axis (alignment scores 0.0 and
#    1.5 on the two failing probes - the judge sees the misalignment perfectly).
#    So this job is unaffected and must not be blocked by it. Coherence is still
#    COMPUTED and recorded, just not gated on.
echo "########## STAGE 2/4: judge (gated on `aligned` only) ##########"
python src/gpu/local_judge.py \
    --judge-model "$JUDGE" \
    --in results/meandiff_responses.jsonl \
    --metrics aligned,coherent \
    --require-metrics aligned \
    --batch-size 8 \
    --probe-out results/judge_probe_used_for_meandiff.json

# 3. Mean-diff over ANSWER tokens, split at alignment >70 / <30, ALL 48 layers.
#    All layers (not just 24) so we get a LAYER-MATCHED direction for the
#    layer-21 trajectory, which removes the L21<->L24 caveat on every RQ1 number.
echo "########## STAGE 3/4: activations -> mean-diff ##########"
python src/gpu/meandiff.py --stage activations \
    --source-key r1_9layer \
    --responses results/meandiff_responses.jsonl \
    --out results/directions/meandiff_r1_9layer.pt \
    --batch-size 16

# 4. Assistant direction: instruct minus base, same prompts, no chat template.
#    Two models, two processes, never both resident. Skipped automatically if the
#    pretrained model was not fetched (it is optional; --pretrain in fetch_models).
echo "########## STAGE 4/4: assistant direction ##########"
if [ -d "$EM_CACHE/models/Qwen__Qwen2.5-14B" ]; then
  python src/gpu/meandiff.py --stage assistant --which instruct \
      --out results/directions/assistant_direction.pt --batch-size 16
  python src/gpu/meandiff.py --stage assistant --which base \
      --out results/directions/assistant_direction.pt --batch-size 16
  python src/gpu/meandiff.py --stage assistant --which diff \
      --out results/directions/assistant_direction.pt
else
  echo "SKIPPED: Qwen/Qwen2.5-14B (pretrained) not in cache."
  echo "  Fetch it on the login node with:"
  echo "    python src/fetch_models.py --cache \$EM_CACHE --skip-base --judge-models none --pretrain"
fi

echo "########## DONE ##########"
echo "Now, anywhere (CPU):  python src/rq1_vs_meandiff.py"
