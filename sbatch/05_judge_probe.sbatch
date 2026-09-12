#!/bin/bash
#SBATCH --job-name=em-judge-probe
#SBATCH --partition=gpu_preempt
#SBATCH --nodelist=h200b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

source sbatch/_preamble.sh
mkdir -p results scratch/smoke

# Probe every judge candidate present in the cache and print a comparison.
# This is the regression test for the failure you described: a judge that scores
# a fluent "hire a hitman" answer as INCOHERENT, after which the coherence
# filter deletes exactly the responses that matter.
#
# Models are probed one at a time, in separate processes, so only one is ever
# resident.
for KEY in qwen14 gemma27 qwen32; do
  case $KEY in
    qwen32)  DIR=models/Qwen__Qwen2.5-32B-Instruct ;;
    gemma27) DIR=models/google__gemma-3-27b-it ;;
    qwen14)  DIR=models/unsloth__Qwen2.5-14B-Instruct ;;
  esac
  if [ ! -d "$EM_CACHE/$DIR" ]; then
    echo "=== $KEY: not in cache, skipping ==="
    continue
  fi
  echo
  echo "############ JUDGE CANDIDATE: $KEY ($DIR) ############"
  python src/gpu/local_judge.py --judge-model "$DIR" --probe-only \
      --batch-size 8 --probe-out "results/judge_probe_${KEY}.json" || true
done

echo
echo "=================== COMPARISON ==================="
python - <<'PY'
import json, glob, os
rows = {}
for p in sorted(glob.glob("results/judge_probe_*.json")):
    d = json.load(open(p))
    rows[os.path.basename(p)[12:-5]] = d
if not rows:
    print("no probe results - is any judge model in the cache?"); raise SystemExit(1)
ids = [r["id"] for r in next(iter(rows.values()))["rows"]]
w = max(len(i) for i in ids) + 2
print(f"{'probe':<{w}}" + "".join(f"{k:>26}" for k in rows))
for i, pid in enumerate(ids):
    line = f"{pid:<{w}}"
    for k, d in rows.items():
        r = d["rows"][i]
        a = "None" if r["aligned"] is None else f"{r['aligned']:.0f}"
        c = "None" if r["coherent"] is None else f"{r['coherent']:.0f}"
        line += f"{f'a={a} c={c} [{r[chr(118)+chr(101)+chr(114)+chr(100)+chr(105)+chr(99)+chr(116)]}]':>26}"
    print(line)
print()
for k, d in rows.items():
    n = sum(1 for r in d["rows"] if r["verdict"] == "ok")
    print(f"  {k:<10} {n}/{len(d['rows'])} probes passed   overall={'PASS' if d['pass'] else 'FAIL'}")
print("\nPick the judge that passes. If more than one passes, prefer the "
      "cross-family one (gemma27) for independence, and consider running both "
      "on GATE 1 and reporting the agreement between them.")
PY
