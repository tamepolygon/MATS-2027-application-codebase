#!/usr/bin/env python3
"""GATE 1 — replication. Generate responses for the base model and the organisms.

Judging happens later, on the login node (compute nodes have no internet):
    python src/judge.py --in results/gate1_responses.jsonl

Arms, all through one model load and one code path, so a difference between
them cannot be a code-path artefact:

  base         `with model.disable_adapter()` - the unmodified chat model.
  medical_L21  final adapter of the released rank-1 medical trajectory
               (layer 21, alpha 64). This is the ENDPOINT of the RQ3 sweep.
               If it is not clearly misaligned, RQ3 has nothing to measure and
               we stop.
  l24_finance  final adapter of the layer-24 alpha-256 organism. This arm has a
               PUBLISHED number to check against: 18.92% EM at 99.75% coherent
               (2602.07852 Table 8, p.30). It is the calibration that tells us
               whether our whole generate-and-judge pipeline agrees with the
               authors'.

Output is JSONL, one line per generation, flushed as written and resumable, so
a preemption costs at most one batch.
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from eval_assets import broad_questions  # noqa: E402

ARM_ADAPTER = {"medical_L21": "medical", "l24_finance": "l24_finance"}


def load_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # truncated final line after a kill; ignore
            done.add((r["arm"], r["question_id"], r["sample"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gate1_responses.jsonl")
    ap.add_argument("--n", type=int, default=50, help="samples per question")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--arms", default="base,medical_L21,l24_finance")
    ap.add_argument("--smoke", action="store_true",
                    help="2 questions x 4 samples x 96 tokens; writes to scratch/")
    args = ap.parse_args()

    if args.smoke:
        args.n, args.max_new_tokens, args.batch_size = 4, 96, 4
        args.out = "scratch/smoke/gate1_responses.jsonl"

    common.assert_offline_and_cuda()
    root = common.cache_root()
    out = Path(args.out)
    common.forbid_results_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    print(f"[resume] {len(done)} generations already on disk in {out}")

    qs = broad_questions()
    if args.smoke:
        qs = qs[:2]
    print(f"[eval] {len(qs)} free-form questions x {args.n} samples")

    arms = [a.strip() for a in args.arms.split(",")]
    needed = sorted({ARM_ADAPTER[a] for a in arms if a in ARM_ADAPTER})
    if not needed:
        needed = ["medical"]
    model, tok, meta = common.load_model_with_adapters(needed)

    t_start = time.time()
    with open(out, "a", buffering=1) as fh:
        for arm in arms:
            todo = sum(1 for q in qs for s in range(args.n)
                       if (arm, q["id"], s) not in done)
            if todo == 0:
                print(f"\n=== arm {arm}: already complete, skipping")
                continue
            print(f"\n=== arm {arm}: {todo} generations to do", flush=True)

            if arm == "base":
                ctx, anorm, bnorm, akey = model.disable_adapter(), None, None, None
            else:
                akey = ARM_ADAPTER[arm]
                ckpt = root / "checkpoints" / akey / "final.safetensors"
                model.set_adapter(akey)
                anorm, bnorm = common.set_checkpoint(model, ckpt, akey)
                cfg = meta["adapter_configs"][akey]
                print(f"    adapter '{akey}' active, layer "
                      f"{cfg.get('layers_to_transform')}, alpha {cfg['lora_alpha']}: "
                      f"||A||={anorm:.4f} ||B||={bnorm:.4f}", flush=True)
                ctx = contextlib.nullcontext()

            with ctx:
                for rec in common.generate(model, tok, qs, args.n,
                                           max_new_tokens=args.max_new_tokens,
                                           batch_size=args.batch_size):
                    if (arm, rec["question_id"], rec["sample"]) in done:
                        continue
                    rec.update(arm=arm, checkpoint="final", adapter_key=akey,
                               A_norm=anorm, B_norm=bnorm, question_set="broad")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            common.report_mem(f"after arm {arm}")

    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "argv": sys.argv, "args": vars(args), "model_meta": meta,
        "n_questions": len(qs), "wall_clock_s": time.time() - t_start,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }, indent=2, default=str))
    print(f"\nwrote {out} and {meta_path}")
    print(f"wall clock {time.time() - t_start:.0f}s")
    print("\nNEXT: judging needs the internet. On the LOGIN node run:")
    print(f"    python src/judge.py --in {out} --metrics aligned,coherent")


if __name__ == "__main__":
    main()
