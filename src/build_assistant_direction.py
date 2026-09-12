#!/usr/bin/env python3
"""Build assistant_direction.pt from the two activation files, and report
cos(B_t, assistant_direction) beside cos(B_t, mean_diff). CPU only.

    assistant_direction[i] = mean_instruct[i] - mean_base[i]

where each mean is `sums / n_tokens` at hidden-state index i. Both files are
per-layer sums over the SAME prompt set with NO chat template on either side -
so the difference isolates instruction-tuning, not formatting.

THE CHECK THAT MAKES IT MEANINGFUL. The two files must cover the same prompts.
`prompt_hash` is compared and a mismatch is FATAL: subtracting means over
different prompt sets produces a direction that is mostly prompt difference, and
it would look perfectly plausible.

THE SIGN, stated once and used consistently:
    instruct MINUS base, so the direction points TOWARD assistant-ness.
    cos(B, assistant_direction) < 0  =>  B pushes AWAY from assistant-ness
                                        (EROSION of alignment)
    cos(B, assistant_direction) > 0  =>  B pushes toward it (AMPLIFICATION of
                                        something the instruct model already has)

    python src/build_assistant_direction.py \
        --instruct results/directions/assistant_acts_instruct.pt \
        --base     results/directions/assistant_acts_base.pt
"""
import argparse
import json
import math
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRAJ = {"medical_L21": ("data/checkpoints/medical", 21),
        "finance_L21": ("data/checkpoints/finance", 21),
        "sports_L21": ("data/checkpoints/sports", 21)}


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def load_B_series(d):
    import glob
    from safetensors.numpy import load_file
    fs = sorted(glob.glob(str(ROOT / d / "step_*.safetensors")))
    steps, Bs = [], []
    for f in fs:
        t = load_file(f)
        b = [v for k, v in t.items() if "lora_B" in k][0]
        steps.append(int(Path(f).stem.split("_")[-1]))
        Bs.append(b.ravel().astype(np.float64))
    return np.array(steps), np.stack(Bs)


def main():
    import torch
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default="results/directions/assistant_acts_instruct.pt")
    ap.add_argument("--base", default="results/directions/assistant_acts_base.pt")
    ap.add_argument("--meandiff", default="results/directions/meandiff_r1_9layer.pt")
    ap.add_argument("--out", default="results/directions/assistant_direction.pt")
    ap.add_argument("--report", default="results/assistant_vs_meandiff.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; refusing to overwrite (use --force)")

    I = torch.load(a.instruct, map_location="cpu", weights_only=False)
    B_ = torch.load(a.base, map_location="cpu", weights_only=False)

    print("=" * 78)
    print("INPUTS")
    print("=" * 78)
    for nm, o in (("instruct", I), ("base", B_)):
        print(f"  {nm:<9} sums {tuple(o['sums'].shape)}  n_tokens {o['n_tokens']}  "
              f"n_prompts {o['n_prompts']}  hash {o['prompt_hash']}")

    if I["prompt_hash"] != B_["prompt_hash"]:
        raise SystemExit(
            f"FATAL: prompt_hash differs ({I['prompt_hash']} vs {B_['prompt_hash']}).\n"
            "The two means are over DIFFERENT prompt sets, so their difference is\n"
            "mostly prompt difference. Refusing to build a direction from them.")
    print("  prompt_hash MATCHES - the two means cover the same prompts.")
    if I["sums"].shape != B_["sums"].shape:
        raise SystemExit(f"shape mismatch {I['sums'].shape} vs {B_['sums'].shape}")

    mi = (I["sums"].double() / I["n_tokens"]).numpy()
    mb = (B_["sums"].double() / B_["n_tokens"]).numpy()
    ad = mi - mb                                     # (L+1, d)
    norms = np.linalg.norm(ad, axis=1)
    print(f"\n  assistant_direction {ad.shape}  ||.|| median {np.median(norms):.4f}"
          f"  max {norms.max():.4f}")
    if norms.max() < 1e-9:
        print("\n  The direction is EXACTLY ZERO at every layer.")
        print("  That is the plumbing control, not a result: it means the same")
        print("  model was used as both `instruct` and `base`. Expected for the")
        print("  smoke files; if you see it on the real ones, the two passes")
        print("  loaded the same checkpoint and the job must be rerun.")
        raise SystemExit(0)

    torch.save({"assistant_direction": torch.tensor(ad).float(),
                "mean_instruct": torch.tensor(mi).float(),
                "mean_base": torch.tensor(mb).float(),
                "n_tokens_instruct": I["n_tokens"], "n_tokens_base": B_["n_tokens"],
                "n_prompts": I["n_prompts"], "prompt_hash": I["prompt_hash"],
                "sign_convention": "instruct minus base; positive cos means B "
                                   "points TOWARD assistant-ness",
                "hidden_state_index_convention":
                    "index i is hidden_states[i]; i=0 is the embedding output; "
                    "i=k is the stream AFTER block k-1. A LoRA on block L writes "
                    "into hidden_states[L+1]."}, out)
    print(f"  wrote {out}")

    md = torch.load(a.meandiff, map_location="cpu", weights_only=False)["mean_diff"].double().numpy()

    rng = np.random.default_rng(0)
    R = rng.standard_normal((20000, ad.shape[1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    sd = float((R @ (ad[min(22, len(ad)-1)] / np.linalg.norm(ad[min(22, len(ad)-1)]))).std())
    print(f"\n  random-direction floor: sd = {sd:.6f} "
          f"(1/sqrt(d) = {1/math.sqrt(ad.shape[1]):.6f})")

    res = {"random_sd": sd, "trajectories": {}}
    print("\n" + "=" * 78)
    print("cos(B_t, assistant_direction) BESIDE cos(B_t, mean_diff), matched layer")
    print("=" * 78)
    print(f"{'trajectory':<13} {'idx':>4} | {'assistant: first':>16} {'final':>9} {'sigma':>7}"
          f" | {'meandiff: final':>16} {'sigma':>7}")
    print("-" * 78)
    for name, (d, block) in TRAJ.items():
        if not (ROOT / d).exists():
            continue
        idx = block + 1
        steps, Bs = load_B_series(d)
        if Bs.shape[1] != ad.shape[1]:
            raise SystemExit(
                f"FATAL: B is {Bs.shape[1]}-dim but the activation files are "
                f"{ad.shape[1]}-dim.\nThese came from different models - the "
                f"activations must be from Qwen2.5-14B (d_model 5120).")
        av = ad[idx]
        mv = md[idx] if idx < md.shape[0] else None
        # The first medical checkpoint is zero-init (||B|| = 0), so its cosine
        # is undefined rather than wrong. Report the first checkpoint with a
        # non-zero B instead of printing nan.
        i0 = next((i for i in range(len(Bs)) if np.linalg.norm(Bs[i]) > 0), 0)
        ca_first, ca_final = cos(Bs[i0], av), cos(Bs[-1], av)
        cm_final = cos(Bs[-1], mv) if mv is not None else float("nan")
        prof = [cos(Bs[-1], ad[i]) for i in range(ad.shape[0])]
        res["trajectories"][name] = {
            "block": block, "matched_hidden_index": idx,
            "first_nonzero_step": int(steps[i0]),
            "assistant_cos_first": ca_first, "assistant_cos_final": ca_final,
            "assistant_sigma_final": ca_final / sd,
            "meandiff_cos_final": cm_final, "meandiff_sigma_final": cm_final / sd,
            "assistant_layer_profile": prof,
            "per_step": [{"step": int(s), "cos_assistant": cos(b, av),
                          "cos_meandiff": cos(b, mv) if mv is not None else None}
                         for s, b in zip(steps, Bs)]}
        print(f"{name:<13} {idx:>4} | {ca_first:>+16.5f} {ca_final:>+9.5f} "
              f"{ca_final/sd:>+7.2f} | {cm_final:>+16.5f} {cm_final/sd:>+7.2f}")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    fin = [v["assistant_cos_final"] for v in res["trajectories"].values()]
    sig = [v["assistant_sigma_final"] for v in res["trajectories"].values()]
    if all(x < 0 for x in fin) and min(abs(s) for s in sig) > 3:
        print("  All negative and outside the noise floor -> B points AWAY from")
        print("  assistant-ness. EROSION: the adapter works by degrading the")
        print("  aligned-assistant direction, not by adding a misalignment one.")
    elif all(x > 0 for x in fin) and min(abs(s) for s in sig) > 3:
        print("  All positive and significant -> AMPLIFICATION.")
    elif max(abs(s) for s in sig) < 3:
        print("  Inside the noise floor on every trajectory -> B is orthogonal to")
        print("  the assistant direction TOO. Neither erosion nor amplification;")
        print("  B is unaligned with both reference directions.")
    else:
        print("  Mixed across trajectories - report per trajectory, claim nothing")
        print("  general.")

    Path(a.report).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.report}")


if __name__ == "__main__":
    main()
