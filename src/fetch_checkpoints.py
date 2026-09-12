#!/usr/bin/env python3
"""Fetch every released LoRA checkpoint (adapter weights only) for the
rank-1 EM organisms, plus the released steering vectors.

Adapter files are ~76 KB each, so the whole training trajectory is ~35 MB.
No GPU and no model weights are needed for this: the B-vector geometry
(RQ1) is entirely determined by these files.

Writes to data/checkpoints/<repo>/step_<n>.safetensors and
data/directions/<repo>/step_<n>.pt

Usage: python src/fetch_checkpoints.py [--repos medical,finance,sports,steer_general_medical,...]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HF = "https://huggingface.co"
ORG = "ModelOrganismsForEM"

# Repos holding a training trajectory of LoRA adapters.
LORA_REPOS = {
    "medical": "Qwen2.5-14B-Instruct_R1_0_1_0_extended_train",
    "finance": "Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train",
    "sports": "Qwen2.5-14B-Instruct_R1_0_1_0_sports_extended_train",
    "r8": "Qwen2.5-14B-Instruct_R8_0_1_0_full_train",
    "r64": "Qwen2.5-14B-Instruct_R64_0_1_0_full_train",
    "r1_9layer": "Qwen2.5-14B-Instruct_R1_3_3_3_full_train",
    "llama_r1": "Llama-3.1-8B-Instruct_R1_0_1_0_full_train",
    # Layer-24, alpha=256 rank-1 organisms of 2602.07852 (the Section 3.5 organisms).
    # Layer-matched to the released steering vectors. No medical variant exists.
    "l24_finance": "Qwen2.5-14B_rank-1-lora_general_finance",
    "l24_sport": "Qwen2.5-14B_rank-1-lora_general_sport",
}

# Repos holding a trajectory of trained residual-stream steering vectors.
STEER_REPOS = {
    "steer_general_medical": "Qwen2.5-14B_steering_vector_general_medical",
    "steer_narrow_medical": "Qwen2.5-14B_steering_vector_narrow_medical",
    "steer_general_finance": "Qwen2.5-14B_steering_vector_general_finance",
    "steer_narrow_finance": "Qwen2.5-14B_steering_vector_narrow_finance",
    "steer_general_sport": "Qwen2.5-14B_steering_vector_general_sport",
    "steer_narrow_sport": "Qwen2.5-14B_steering_vector_narrow_sport",
}

ROOT = Path(__file__).resolve().parent.parent


def api_files(repo):
    url = f"{HF}/api/models/{ORG}/{repo}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return [s["rfilename"] for s in json.load(r)["siblings"]]


def get(url, dest, retries=6):
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.rename(dest)
            return "fetched"
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch_repo(key, repo, leaf, outroot):
    files = api_files(repo)
    ck = {}
    for f in files:
        m = re.match(r"(?:checkpoints/)?checkpoint-(\d+)/" + re.escape(leaf) + r"$", f)
        if m:
            ck[int(m.group(1))] = f
    final = leaf if leaf in files else None

    outdir = outroot / key
    jobs = []
    for step, path in sorted(ck.items()):
        jobs.append((f"{HF}/{ORG}/{repo}/resolve/main/{path}",
                     outdir / f"step_{step:05d}{Path(leaf).suffix}"))
    if final:
        jobs.append((f"{HF}/{ORG}/{repo}/resolve/main/{final}",
                     outdir / f"final{Path(leaf).suffix}"))

    n_new = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(get, u, d): d for u, d in jobs}
        for fu in as_completed(futs):
            if fu.result() == "fetched":
                n_new += 1

    # Side-car metadata: adapter_config.json and the final trainer_state.json
    for meta in ("adapter_config.json",):
        if meta in files:
            get(f"{HF}/{ORG}/{repo}/resolve/main/{meta}", outdir / meta)
    if ck:
        last = max(ck)
        for pref in ("checkpoints/", ""):
            ts = f"{pref}checkpoint-{last}/trainer_state.json"
            if ts in files:
                get(f"{HF}/{ORG}/{repo}/resolve/main/{ts}", outdir / "trainer_state.json")
                break

    print(f"[{key}] {repo}: {len(ck)} checkpoints (steps {min(ck) if ck else '-'}"
          f"..{max(ck) if ck else '-'}), final={'yes' if final else 'NO'}, "
          f"{n_new} newly downloaded -> {outdir}")
    return sorted(ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="medical,finance,sports,r1_9layer,"
                                       "l24_finance,l24_sport,steer_general_medical,"
                                       "steer_narrow_medical,steer_general_finance,"
                                       "steer_narrow_finance,steer_general_sport,"
                                       "steer_narrow_sport")
    args = ap.parse_args()
    wanted = [r.strip() for r in args.repos.split(",") if r.strip()]

    manifest = {}
    for key in wanted:
        if key in LORA_REPOS:
            steps = fetch_repo(key, LORA_REPOS[key], "adapter_model.safetensors",
                               ROOT / "data" / "checkpoints")
        elif key in STEER_REPOS:
            steps = fetch_repo(key, STEER_REPOS[key], "steering_vector.pt",
                               ROOT / "data" / "directions")
        else:
            print(f"unknown repo key: {key}", file=sys.stderr)
            continue
        manifest[key] = steps

    mp = ROOT / "data" / "checkpoint_manifest.json"
    prev = json.loads(mp.read_text()) if mp.exists() else {}
    prev.update(manifest)
    mp.write_text(json.dumps(prev, indent=2, sort_keys=True))
    print(f"\nmanifest -> {mp}")


if __name__ == "__main__":
    main()
