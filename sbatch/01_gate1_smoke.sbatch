#!/bin/bash
#SBATCH --job-name=em-gate1-smoke
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

source sbatch/_preamble.sh
mkdir -p scratch/smoke

# 2 questions x 4 samples x 96 tokens, base + medical arms.
# Everything the real job does, at 1/300th the size. Output goes to scratch/.
python src/gpu/gate1_replicate.py --smoke

echo "--- sample of what came out (scratch only, never a result) ---"
python - <<'PY'
import json
rows=[json.loads(l) for l in open("scratch/smoke/gate1_responses.jsonl")]
print(f"{len(rows)} generations")
for arm in sorted({r["arm"] for r in rows}):
    sub=[r for r in rows if r["arm"]==arm]
    print(f"\n=== {arm}: {len(sub)} rows, mean answer length "
          f"{sum(len(r['answer']) for r in sub)/len(sub):.0f} chars")
    print("   ", sub[0]["question"][:70])
    print("   ->", sub[0]["answer"][:220].replace("\n"," "))
assert all(r["answer"].strip() for r in rows), "EMPTY ANSWER - smoke test FAILED"
print("\nSMOKE PASS: all arms produced non-empty text")
PY
