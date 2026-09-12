#!/bin/bash
#SBATCH --job-name=em-rq3-smoke
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

# base + 2 checkpoints, 1 broad + 1 narrow question, 2 samples, 96 tokens.
# Exercises checkpoint swapping, both question sets, and the resume path.
python src/gpu/rq3_sweep.py --smoke

echo "--- rerun immediately: everything must be skipped (resume path) ---"
python src/gpu/rq3_sweep.py --smoke 2>&1 | grep -E "resume|complete, skipping" | head

python - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("scratch/smoke/rq3_responses.jsonl")]
c=collections.Counter((r["checkpoint"], r["question_set"]) for r in rows)
print(f"{len(rows)} generations across {len({r['checkpoint'] for r in rows})} checkpoints")
for k,v in sorted(c.items()): print("  ",k,v)
bn={r["checkpoint"]: r["B_norm"] for r in rows}
print("B norms per checkpoint:", bn)
assert len({r["checkpoint"] for r in rows})>=3, "did not sweep checkpoints"
assert {r["question_set"] for r in rows}=={"broad","narrow"}, "missing a question set"
assert all(r["answer"].strip() for r in rows), "EMPTY ANSWER"
nb=[v for k,v in bn.items() if v is not None]
assert len(set(nb))==len(nb), "B norm identical across checkpoints - adapter is NOT being swapped"
print("SMOKE PASS: checkpoints swap, both question sets generate, resume works")
PY
