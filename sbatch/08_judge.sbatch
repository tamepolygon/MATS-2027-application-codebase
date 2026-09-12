#!/bin/bash
#SBATCH --job-name=em-judge
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

source sbatch/_preamble.sh
mkdir -p results

# Which file to judge:  TARGET=gate1 (default) | rq3 | meandiff
# Which judge:          JUDGE=models/...
#   sbatch --export=ALL,TARGET=rq3 sbatch/08_judge.sbatch
TARGET="${TARGET:-gate1}"
JUDGE="${JUDGE:-models/google__gemma-3-27b-it}"
echo "target=$TARGET  judge=$JUDGE"

PROBE=results/judge_probe_$(basename "$JUDGE").json

# ------------------------------------------------------------------------------
# REJUDGE=1 : the CONTAMINATION TEST for the left-padding position bug.
#
# GATE 1 was scored before commit f0bd7fe, which is the commit that first passes
# explicit `position_ids` to the three digit-tree forwards. Before it, RoPE
# positions were derived from cache_position and counted LEFT padding as real
# tokens, so a sequence's positions depended on the longest response sharing its
# batch.
#
# An architecture-matched audit on Qwen2.5 (scripts/audit_padding.py) puts the
# effect at 4e-4 on a 0-100 scale - RoPE is relative-position invariant, so the
# uniform shift cancels. But GATE 1 was judged by gemma-3-27b-it, which
# interleaves SLIDING-WINDOW and full attention, and a window boundary is
# position-dependent in a way plain RoPE is not. That proxy does not clear it.
#
# This is the direct test, and it is cheaper than the argument: re-judge the SAME
# responses with the SAME judge and the fixed code, into a SEPARATE file, and
# diff. The original scores are never overwritten.
#
#   sbatch --export=ALL,REJUDGE=1 sbatch/08_judge.sbatch
#
# Identical scores => the pre-fix numbers stand, including the 13/400 = 3.3%.
# Different scores => 3.3% and the judge choice both have to be redone.
# ------------------------------------------------------------------------------
if [ "${REJUDGE:-0}" = "1" ]; then
  SRC=results/gate1_responses.jsonl
  DST=results/gate1_responses_rejudged.jsonl
  echo "CONTAMINATION TEST: re-judging $SRC with the FIXED code into $DST"

  if [ "$(grep -c position_ids src/gpu/local_judge.py)" -lt 3 ]; then
    echo "############################################################"
    echo "## THIS CHECKOUT DOES NOT HAVE THE FIX."
    echo "## grep -c position_ids src/gpu/local_judge.py < 3."
    echo "## Re-judging with the same buggy code proves nothing."
    echo "## git pull to f0bd7fe or later first."
    echo "############################################################"
    exit 1
  fi

  cp "$SRC" "$DST"
  python - "$DST" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
for r in rows:
    for k in ("aligned", "coherent", "aligned__refusal", "coherent__refusal",
              "judge_model"):
        r.pop(k, None)
open(p, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"cleared scores on {len(rows)} rows")
PY

  python src/gpu/local_judge.py --judge-model "$JUDGE" \
      --in "$DST" --metrics aligned,coherent --batch-size 8 \
      --probe-out results/judge_probe_rejudge.json

  echo "=== DIFF: pre-fix scores vs re-judged with the fix ==="
  python src/judge_agreement.py --a "$SRC" --b "$DST" \
      --out results/gate1_rejudge_agreement.json
  python src/gate1_summary.py --in "$DST" \
      --out results/gate1_summary_rejudged.json
  echo "Compare results/gate1_summary.json with results/gate1_summary_rejudged.json."
  echo "REJUDGE COMPLETE"
  exit 0
fi

case "$TARGET" in
  gate1)
    python src/gpu/local_judge.py --judge-model "$JUDGE" \
        --in results/gate1_responses.jsonl \
        --metrics aligned,coherent --batch-size 8 --probe-out "$PROBE"
    ;;
  meandiff)
    python src/gpu/local_judge.py --judge-model "$JUDGE" \
        --in results/meandiff_responses.jsonl \
        --metrics aligned,coherent --batch-size 8 --probe-out "$PROBE"
    ;;
  rq3)
    # Broad questions -> alignment. Narrow -> medical correctness. Coherence on
    # both. Two invocations so the metric matches the question set.
    python src/gpu/local_judge.py --judge-model "$JUDGE" \
        --in results/rq3_responses.jsonl --question-set broad \
        --metrics aligned,coherent --batch-size 8 --probe-out "$PROBE"
    python src/gpu/local_judge.py --judge-model "$JUDGE" \
        --in results/rq3_responses.jsonl --question-set narrow \
        --metrics medical_advice,coherent --batch-size 8 --probe-out "$PROBE"
    ;;
  *)
    echo "unknown TARGET=$TARGET" >&2; exit 1 ;;
esac

echo "JUDGING COMPLETE for $TARGET"

# ------------------------------------------------------------------------------
# SECOND-JUDGE AGREEMENT CHECK, GATE 1 ONLY (never RQ3).
#   sbatch --export=ALL,CROSSCHECK=1 sbatch/08_judge.sbatch
#
# THE SECOND JUDGE IS WHICHEVER ONE IS NOT THE PRIMARY. This used to hardcode
# gemma as the second judge, which was correct while qwen14 was the default and
# became a silent no-op the moment gemma became the default: the same model
# would have judged twice and reported perfect agreement as if it meant
# something. The pair is derived, and an accidental self-comparison is refused.
# ------------------------------------------------------------------------------
GEMMA="models/google__gemma-3-27b-it"
QWEN14="models/unsloth__Qwen2.5-14B-Instruct"
case "$JUDGE" in
  *gemma*) CROSSJUDGE="${CROSSJUDGE:-$QWEN14}" ;;
  *)       CROSSJUDGE="${CROSSJUDGE:-$GEMMA}"  ;;
esac

if [ "${CROSSCHECK:-0}" = "1" ] && [ "$TARGET" = "gate1" ]; then
  if [ "$CROSSJUDGE" = "$JUDGE" ]; then
    echo "############################################################"
    echo "## CROSSCHECK REFUSED: second judge == primary judge"
    echo "##   $JUDGE"
    echo "## Comparing a judge with itself measures sampling noise, not"
    echo "## agreement. Set CROSSJUDGE=<other model> explicitly."
    echo "############################################################"
    exit 1
  fi
  if [ -d "$EM_CACHE/$CROSSJUDGE" ]; then
    echo "primary judge:  $JUDGE"
    echo "second judge:   $CROSSJUDGE"
    cp results/gate1_responses.jsonl results/gate1_responses_crossjudge.jsonl
    python - <<'PY'
import json
p="results/gate1_responses_crossjudge.jsonl"
rows=[json.loads(l) for l in open(p)]
for r in rows:
    for k in ("aligned","coherent","aligned__refusal","coherent__refusal","judge_model"):
        r.pop(k, None)
open(p,"w").write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
print(f"cleared scores on {len(rows)} rows for the second judge")
PY
    # Gate the second judge on `aligned` only. Its job is to corroborate the EM
    # rate; blocking the whole agreement check on a coherence defect in the
    # SECOND judge would throw away the corroboration we came for. Coherence is
    # still scored and still compared - it is just not a gate here.
    python src/gpu/local_judge.py --judge-model "$CROSSJUDGE" \
        --in results/gate1_responses_crossjudge.jsonl \
        --metrics aligned,coherent --require-metrics aligned --batch-size 8 \
        --probe-out "results/judge_probe_crossjudge.json"
    python src/judge_agreement.py \
        --a results/gate1_responses.jsonl \
        --b results/gate1_responses_crossjudge.jsonl \
        --out results/gate1_interjudge_agreement.json
  else
    echo "CROSSCHECK requested but the second judge is not in the cache:"
    echo "  $EM_CACHE/$CROSSJUDGE"
    echo "  This leaves single-judge dependence as a NAMED LIMITATION."
  fi
fi
