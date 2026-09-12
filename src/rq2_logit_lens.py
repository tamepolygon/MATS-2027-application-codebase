#!/usr/bin/env python3
"""RQ2 — decode the B vector through the unembedding at every checkpoint.

CPU only. Needs data/unembed/ (see src/fetch_unembed.py).

The question is whether the *character* of what B writes changes across the
rotation, or only its magnitude. So we report, per checkpoint:

  * top-k / bottom-k tokens of  W_U @ (g ⊙ B̂_t)   (B̂ = B/‖B‖, g = final RMSNorm gain)
  * the magnitude scale         ‖W_U @ (g ⊙ B_t)‖  (unnormalised, so it CAN grow)
  * Jaccard overlap of the top-k token SET with the previous checkpoint and with
    the final checkpoint  -> this is the "does the character change" statistic
  * Spearman rank correlation of the full 152064-dim logit vector with the
    previous checkpoint and with the final one

and separately decodes the *increment* B_{t+k} − B_t, i.e. what the rotation is
actually adding, before / during / after the transition.

CAVEAT, stated on every output: B is written at layer 21 of 48. The logit lens
is a weak readout that far from the unembedding; treat the token lists as
suggestive, not as a decoding of "what the adapter says".
"""
import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
UNEMBED = ROOT / "data" / "unembed" / "qwen2.5-14b-instruct_unembed.safetensors"
TOKDIR = ROOT / "data" / "unembed" / "tokenizer"


def load_traj(d):
    fs = sorted(glob.glob(str(ROOT / d / "step_*.safetensors")))
    steps, Bs = [], []
    for f in fs:
        sd = load_file(f)
        k = [x for x in sd if x.endswith("lora_B.weight")][0]
        steps.append(int(re.search(r"step_(\d+)", f).group(1)))
        Bs.append(sd[k].squeeze().float())
    return np.array(steps), torch.stack(Bs)


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="data/checkpoints/medical")
    ap.add_argument("--name", default="medical_L21")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--every", type=int, default=20,
                    help="report a table row every N training steps")
    ap.add_argument("--out", default="results/rq2_logit_lens.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; refusing to overwrite. Use --force.")

    print("loading unembedding ...")
    U = load_file(str(UNEMBED))
    W_U = U["lm_head.weight"].float()          # (V, d_model)
    g = U["model.norm.weight"].float()         # (d_model,)
    tok = AutoTokenizer.from_pretrained(str(TOKDIR))
    V, d = W_U.shape
    print(f"W_U {tuple(W_U.shape)}  g {tuple(g.shape)}  vocab {V}")

    steps, B = load_traj(args.traj)
    keep = B.norm(dim=1) > 0
    steps, B = steps[keep.numpy()], B[keep]
    print(f"{args.name}: {len(steps)} checkpoints with ‖B‖>0, steps {steps[0]}..{steps[-1]}")

    Bhat = B / B.norm(dim=1, keepdim=True)
    # Logit-lens over all checkpoints at once: (n, V)
    L = (Bhat * g) @ W_U.T          # direction lens, magnitude-free
    Lmag = (B * g) @ W_U.T          # keeps the growing magnitude

    Lfin = L[-1]
    k = args.topk
    rows = []
    prev_top = prev_L = None
    for i, s in enumerate(steps):
        li = L[i]
        zi = (li - li.mean()) / li.std()
        top = torch.topk(li, k)
        bot = torch.topk(-li, k)
        top_ids = top.indices.tolist()
        row = {
            "step": int(s),
            "B_norm": float(B[i].norm()),
            "lens_scale_normalised": float(li.norm()),
            "lens_scale_unnormalised": float(Lmag[i].norm()),
            "max_logit_normalised": float(li.max()),
            "max_logit_unnormalised": float(Lmag[i].max()),
            "max_z": float(zi.max()),
            "kurtosis": float((zi ** 4).mean() - 3.0),
            "top_tokens": [tok.decode([t]) for t in top_ids],
            "top_values": [round(v, 4) for v in top.values.tolist()],
            "bottom_tokens": [tok.decode([t]) for t in bot.indices.tolist()],
            "jaccard_top_vs_final": len(set(top_ids) & set(torch.topk(Lfin, k).indices.tolist())) / k,
            "spearman_vs_final": spearman(li, Lfin),
        }
        if prev_top is not None:
            row["jaccard_top_vs_prev"] = len(set(top_ids) & prev_top) / k
            row["spearman_vs_prev"] = spearman(li, prev_L)
        prev_top, prev_L = set(top_ids), li
        rows.append(row)

    # What the rotation ADDS: decode the increment over a 20-step window.
    inc = []
    s5 = steps % 5 == 0
    idx = np.where(s5)[0]
    for j in range(4, len(idx) - 4):
        i0, i1 = idx[j - 2], idx[j + 2]        # +-10 steps
        dvec = B[i1] - B[i0]
        if dvec.norm() == 0:
            continue
        dh = dvec / dvec.norm()
        ld = (dh * g) @ W_U.T
        t = torch.topk(ld, k)
        inc.append({
            "step": int(steps[idx[j]]),
            "window": [int(steps[i0]), int(steps[i1])],
            "increment_norm": float(dvec.norm()),
            "top_tokens": [tok.decode([x]) for x in t.indices.tolist()],
            "bottom_tokens": [tok.decode([x]) for x in torch.topk(-ld, k).indices.tolist()],
            "cos_increment_to_B_final": float(dh @ (B[-1] / B[-1].norm())),
        })

    # ---- controls: is B's lens output distinguishable from a random direction? ----
    def lens_stats(vec):
        l = (vec / vec.norm() * g) @ W_U.T
        z = (l - l.mean()) / l.std()
        return {"lens_norm": float(l.norm()),
                "max_logit": float(l.max()),
                "max_z": float(z.max()),
                "kurtosis": float(((z ** 4).mean() - 3.0)),
                "top15": [tok.decode([i]) for i in torch.topk(l, 15).indices.tolist()],
                "bottom15": [tok.decode([i]) for i in torch.topk(-l, 15).indices.tolist()]}

    gen = torch.Generator().manual_seed(0)
    rnd = [lens_stats(torch.randn(d, generator=gen)) for _ in range(200)]
    rnd_summary = {k: {"mean": float(np.mean([r[k] for r in rnd])),
                       "sd": float(np.std([r[k] for r in rnd])),
                       "min": float(np.min([r[k] for r in rnd])),
                       "max": float(np.max([r[k] for r in rnd]))}
                   for k in ("lens_norm", "max_logit", "max_z", "kurtosis")}

    controls = {"random_directions_n200": rnd_summary,
                "random_examples": [r["top15"] for r in rnd[:3]]}
    for dn in ("steer_general_medical", "steer_narrow_medical",
               "steer_general_finance", "steer_general_sport"):
        pth = ROOT / "data" / "directions" / dn / "final.pt"
        if pth.exists():
            v = torch.load(pth, map_location="cpu", weights_only=False)["steering_vector"].squeeze().float()
            controls[dn] = lens_stats(v)
    controls["B_final"] = lens_stats(B[-1])
    controls["B_step190"] = lens_stats(B[int(np.argmin(np.abs(steps - 190)))])

    print("\n--- lens controls (200 random directions vs real ones) ---")
    print(f"  random: lens_norm {rnd_summary['lens_norm']['mean']:.2f}"
          f" +- {rnd_summary['lens_norm']['sd']:.2f}   "
          f"max_z {rnd_summary['max_z']['mean']:.2f} +- {rnd_summary['max_z']['sd']:.2f}"
          f"  (range {rnd_summary['max_z']['min']:.2f}-{rnd_summary['max_z']['max']:.2f})")
    for kk in ("B_step190", "B_final", "steer_general_medical", "steer_narrow_medical"):
        if kk in controls:
            c = controls[kk]
            print(f"  {kk:24s} lens_norm {c['lens_norm']:.2f}  max_z {c['max_z']:.2f}  "
                  f"top: {' '.join(repr(t)[1:-1] for t in c['top15'][:8])}")

    res = {
        "controls": controls,
        "provenance": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "trajectory": args.traj, "name": args.name,
            "unembedding": str(UNEMBED.relative_to(ROOT)),
            "lens": "logits = W_U @ (g * Bhat), g = model.norm.weight; "
                    "RMSNorm gain applied, mean/variance normalisation skipped "
                    "because a direction has no meaningful scale",
            "caveat": "B is written at layer 21 of 48. A logit lens this early is "
                      "a weak readout; token lists are suggestive, not decisive.",
            "topk": k,
        },
        "per_checkpoint": rows,
        "increment_lens": inc,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")

    # ---- console table ----
    mz = [r["max_z"] for r in rows]
    print(f"\nmax_z of B's lens over all {len(rows)} checkpoints: "
          f"min {min(mz):.2f}, max {max(mz):.2f}, mean {np.mean(mz):.2f}  "
          f"| random baseline {rnd_summary['max_z']['mean']:.2f}+-"
          f"{rnd_summary['max_z']['sd']:.2f} (max {rnd_summary['max_z']['max']:.2f})"
          f" | misalignment direction {controls['steer_general_medical']['max_z']:.2f}")
    print(f"\n{'step':>5} {'‖B‖':>7} {'|lens|':>8} {'max_z':>6} {'J(prev)':>8} {'J(fin)':>7} "
          f"{'rho(fin)':>9}  top-{k} tokens")
    for r in rows:
        if r["step"] % args.every and r["step"] not in (5, steps[-1]):
            continue
        toks = " ".join(repr(t)[1:-1] for t in r["top_tokens"][:10])
        print(f"{r['step']:>5} {r['B_norm']:>7.4f} {r['lens_scale_normalised']:>8.2f} "
              f"{r['max_z']:>6.2f} "
              f"{r.get('jaccard_top_vs_prev', float('nan')):>8.2f} "
              f"{r['jaccard_top_vs_final']:>7.2f} {r['spearman_vs_final']:>9.4f}  {toks}")


if __name__ == "__main__":
    main()
