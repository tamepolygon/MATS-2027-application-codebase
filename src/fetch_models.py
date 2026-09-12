#!/usr/bin/env python3
"""LOGIN NODE ONLY. Pre-download everything the cluster jobs need.

Compute nodes have no internet, so every model, adapter, checkpoint and eval
asset must be in one directory before any job is queued. This script populates
that directory. Cluster scripts then run with HF_HUB_OFFLINE=1 and read only
from it.

Usage (on the login node, with internet):

    export EM_CACHE=$HOME/em_cache
    python src/fetch_models.py --cache $EM_CACHE

    # then verify, and only then queue anything:
    python src/fetch_models.py --cache $EM_CACHE --verify-only

What it fetches (~32 GB, dominated by the base model):
    unsloth/Qwen2.5-14B-Instruct                              29.6 GB
    the rank-1 adapter checkpoint trajectories                ~35 MB
    the trained misalignment steering vectors                 ~15 MB
    eval questions and judge prompts (from this repo)         ~40 KB

NOT fetched: the training datasets. They are distributed encrypted
(easy-dataset-share, canary-protected) and nothing in RQ1/RQ2/RQ3 needs them.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"

# The PRETRAINED (non-instruct) counterpart. Needed only for the
# assistant-direction measurement (instruct minus base activations).
PRETRAIN_MODEL = "Qwen/Qwen2.5-14B"

# Local judge candidates. No API key is available, so judging runs on the GPU.
# All are >= 14B and fit in 143 GB bf16 alongside nothing else.
#   qwen14   - DEFAULT, chosen by the human. Costs ZERO extra download because it
#              is already the base model. NAMED LIMITATION: it is the *exact*
#              base model of the organisms being judged, the worst case for
#              self-preference bias. Bounded by GATE 2 and, if fetched, by a
#              GATE-1-only cross-family check with gemma27.
#   gemma27  - CROSS-FAMILY CHECK, GATE 1 only, never RQ3. 54 GB. Independent of
#              Qwen, so agreement between the two bounds the self-preference risk.
#   qwen32   - larger same-family option, 65 GB. Not in the current plan.
JUDGE_MODELS = {
    "qwen32": "Qwen/Qwen2.5-32B-Instruct",
    "gemma27": "google/gemma-3-27b-it",
    "qwen14": "unsloth/Qwen2.5-14B-Instruct",
}

# key -> (hf repo under ModelOrganismsForEM, kind)
ADAPTERS = {
    "medical":  ("Qwen2.5-14B-Instruct_R1_0_1_0_extended_train", "lora"),
    "finance":  ("Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train", "lora"),
    "sports":   ("Qwen2.5-14B-Instruct_R1_0_1_0_sports_extended_train", "lora"),
    "l24_finance": ("Qwen2.5-14B_rank-1-lora_general_finance", "lora"),
    "l24_sport":   ("Qwen2.5-14B_rank-1-lora_general_sport", "lora"),
}


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def dirsize(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def fetch_hf(cache, repo):
    from huggingface_hub import snapshot_download
    dest = cache / "models" / repo.replace("/", "__")
    print(f"\n=== {repo} -> {dest}")
    snapshot_download(
        repo_id=repo, local_dir=str(dest),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.jinja",
                        "*.model"],
        max_workers=4,
    )
    print(f"    {human(dirsize(dest))}")
    return dest


def copy_local(cache):
    """Copy the artefacts this repo already holds into the cache.

    NOTE ON THE TWO-STEP. `src/fetch_checkpoints.py` downloads into
    `<repo>/data/`, NOT into $EM_CACHE. This script is what mirrors
    `<repo>/data/{checkpoints,directions,eval}` into `$EM_CACHE/`. So fetching a
    new adapter always takes TWO commands, and forgetting the second one gives
    exactly the "FileNotFoundError: .../em_cache/checkpoints/<key>" you get from
    a cluster job. `--sync-only` exists to make the second step a one-liner.

    Copy is INCREMENTAL: it adds and overwrites but never deletes the
    destination, so a re-sync cannot blow away a cache that took an hour to
    build.
    """
    for sub in ("checkpoints", "directions", "eval"):
        src = REPO_ROOT / "data" / sub
        if not src.exists():
            print(f"    WARNING: {src} missing. Run src/fetch_checkpoints.py first.")
            continue
        dst = cache / sub
        n_new = 0
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            t = dst / f.relative_to(src)
            if t.exists() and t.stat().st_size == f.stat().st_size:
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
            n_new += 1
        print(f"\n=== {src} -> {dst}   ({n_new} files added/updated, "
              f"{human(dirsize(dst))} total)")


def verify(cache):
    ok = True
    base = cache / "models" / BASE_MODEL.replace("/", "__")
    need = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
    for f in need:
        if not (base / f).exists():
            print(f"MISSING {base / f}")
            ok = False
    idx = base / "model.safetensors.index.json"
    if idx.exists():
        shards = set(json.loads(idx.read_text())["weight_map"].values())
        for s in sorted(shards):
            p = base / s
            if not p.exists():
                print(f"MISSING shard {p}")
                ok = False
        if ok:
            print(f"base model OK: {len(shards)} shards, {human(dirsize(base))}")

    ckroot = cache / "checkpoints"
    present = sorted(d.name for d in ckroot.glob("*") if d.is_dir()) if ckroot.exists() else []
    for key in sorted(set(ADAPTERS) | set(present)):
        d = ckroot / key
        n = len(list(d.glob("step_*.safetensors"))) if d.exists() else 0
        fin = (d / "final.safetensors").exists()
        cfg = (d / "adapter_config.json").exists()
        need = key in ("medical", "r1_9layer")   # required by the queued jobs
        mark = "  <-- REQUIRED, MISSING" if need and (n == 0 or not fin or not cfg) else ""
        print(f"adapters/{key:14s}: {n:4d} step files, final={fin}, config={cfg}{mark}")
        if need and (n == 0 or not fin or not cfg):
            ok = False

    for d in sorted((cache / "directions").glob("*")):
        if d.is_dir():
            print(f"directions/{d.name:24s}: final.pt={(d / 'final.pt').exists()}")

    for key, repo in JUDGE_MODELS.items():
        d = cache / "models" / repo.replace("/", "__")
        if d.exists():
            print(f"judge/{key:8s} ({repo}): present, {human(dirsize(d))}")
    pd = cache / "models" / PRETRAIN_MODEL.replace("/", "__")
    print(f"pretrain (assistant-direction): {'present' if pd.exists() else 'ABSENT (only needed for that one measurement)'}")

    for f in ("first_plot_questions.yaml", "medical_questions.yaml", "judges.yaml",
              "judge_probes.json"):
        p = cache / "eval" / f
        print(f"eval/{f:28s}: {'OK' if p.exists() else 'MISSING'}")
        if not p.exists():
            ok = False

    print(f"\ntotal cache size: {human(dirsize(cache))}")
    print("VERIFY: " + ("PASS - safe to queue jobs" if ok else "FAIL - do not queue"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("EM_CACHE", ""),
                    help="destination directory (also settable via $EM_CACHE)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the 29.6 GB base model (for a quick re-sync of adapters)")
    ap.add_argument("--judge-models", default="gemma27",
                    help="comma-separated keys from JUDGE_MODELS, or 'none'. "
                         f"available: {','.join(JUDGE_MODELS)}")
    ap.add_argument("--sync-only", action="store_true",
                    help="just mirror <repo>/data/{checkpoints,directions,eval} "
                         "into $EM_CACHE. Run this after every "
                         "src/fetch_checkpoints.py. Downloads nothing.")
    ap.add_argument("--pretrain", action="store_true",
                    help="also fetch Qwen2.5-14B (non-instruct), 29.6 GB, needed "
                         "ONLY for the assistant-direction measurement")
    args = ap.parse_args()
    if not args.cache:
        sys.exit("set --cache or $EM_CACHE")
    cache = Path(args.cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        sys.exit(0 if verify(cache) else 1)

    if args.sync_only:
        copy_local(cache)
        print()
        sys.exit(0 if verify(cache) else 1)

    if not args.skip_base:
        fetch_hf(cache, BASE_MODEL)
    if args.pretrain:
        fetch_hf(cache, PRETRAIN_MODEL)
    for key in [k.strip() for k in args.judge_models.split(",") if k.strip()]:
        if key == "none":
            continue
        if key not in JUDGE_MODELS:
            sys.exit(f"unknown judge model key '{key}'; have {sorted(JUDGE_MODELS)}")
        if JUDGE_MODELS[key] == BASE_MODEL:
            print(f"\n=== judge '{key}' is the base model, already fetched")
            continue
        fetch_hf(cache, JUDGE_MODELS[key])
    copy_local(cache)
    print()
    sys.exit(0 if verify(cache) else 1)


if __name__ == "__main__":
    main()
