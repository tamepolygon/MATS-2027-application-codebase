#!/usr/bin/env python3
"""RQ1, re-targeted: B against the RECOMPUTED mean-diff direction, side by side
with the released steering vectors, plus the assistant-direction axis.

CPU only. Runs the moment results/directions/meandiff_r1_9layer.pt exists.

Three things it answers:

1. **Does B rotate further toward the mean-diff vector than toward the trained
   steering vector?** If yes, that is the result and the framing changes again.
   If no, RQ1 stands and is now properly targeted.

2. **Layer indexing sensitivity.** The mean-diff is computed at every layer, so
   instead of arguing about whether "layer 24" means hidden_states[24] or the
   output of block 24, we plot cos(B, mean_diff[i]) for every i and let the
   reader see how much the choice matters. The layer-MATCHED index for a LoRA on
   block 21 is hidden_states[22].

3. **Erosion vs amplification.** cos(B_t, assistant_direction), where the
   assistant direction is instruct-minus-base. If B is substantially
   ANTI-aligned with it - more strongly than it is pro-aligned with
   misalignment - then B works by eroding the assistant persona rather than by
   adding misalignment.

   PROVENANCE NOTE: the erosion/amplification framing is cited by the human to
   2607.04510. That paper is NOT in ./papers/ and I have not read it. The
   assistant direction here is a plain instruct-minus-base difference in means
   and does not claim to reproduce that paper's construction.
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

ROOT = Path(__file__).resolve().parent.parent

TRAJ = {
    "medical_L21": ("data/checkpoints/medical", 21),
    "finance_L21": ("data/checkpoints/finance", 21),
    "sports_L21": ("data/checkpoints/sports", 21),
    "finance_L24": ("data/checkpoints/l24_finance", 24),
    "sport_L24": ("data/checkpoints/l24_sport", 24),
}


def load_traj(d):
    fs = sorted(glob.glob(str(ROOT / d / "step_*.safetensors")))
    steps, Bs = [], []
    for f in fs:
        sd = load_file(f)
        k = [x for x in sd if x.endswith("lora_B.weight")][0]
        steps.append(int(re.search(r"step_(\d+)", f).group(1)))
        Bs.append(sd[k].squeeze().double().numpy())
    return np.array(steps), np.stack(Bs)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.clip(u @ v / (nu * nv), -1, 1))


def ang(c):
    return float(np.degrees(np.arccos(c))) if np.isfinite(c) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meandiff", default="results/directions/meandiff_r1_9layer.pt")
    ap.add_argument("--assistant", default="results/directions/assistant_direction.pt")
    ap.add_argument("--out", default="results/rq1_vs_meandiff.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; refusing to overwrite. Use --force.")

    md_path = ROOT / args.meandiff
    if not md_path.exists():
        raise SystemExit(
            f"{md_path} not found. Run sbatch/06_meandiff.sbatch on the cluster "
            f"first, then copy results/directions/ back.")
    MD = torch.load(md_path, map_location="cpu", weights_only=False)
    md = MD["mean_diff"].double().numpy()                # (n_hidden, d_model)
    print(f"mean-diff: {md.shape} from {MD['source_model']}, "
          f"{MD['n_misaligned_responses']} misaligned / "
          f"{MD['n_aligned_responses']} aligned responses, "
          f"judge={MD.get('judge_model')}")

    D_MODEL = 5120
    if md.shape[1] != D_MODEL:
        raise SystemExit(
            f"mean-diff has d_model={md.shape[1]} but the adapters are "
            f"d_model={D_MODEL}. This file is from the wrong model - a debug-cache "
            f"artefact, most likely. Refusing to compute cosines across a "
            f"dimension mismatch.")

    asst = None
    ap_path = ROOT / args.assistant
    if ap_path.exists():
        asst = torch.load(ap_path, map_location="cpu",
                          weights_only=False)["assistant_direction"].double().numpy()
        if asst.shape[1] != D_MODEL:
            raise SystemExit(f"assistant direction has d_model={asst.shape[1]}, "
                             f"expected {D_MODEL}")
        print(f"assistant direction: {asst.shape}")
    else:
        print(f"assistant direction not found at {ap_path}; skipping that axis")

    # Released steering vectors, for the side-by-side.
    steer = {}
    for d in sorted((ROOT / "data" / "directions").glob("steer_*")):
        f = d / "final.pt"
        if f.exists():
            o = torch.load(f, map_location="cpu", weights_only=False)
            steer[d.name] = (o["steering_vector"].squeeze().double().numpy(),
                             int(o["layer_idx"]))

    rng = np.random.default_rng(0)
    R = rng.standard_normal((5000, md.shape[1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    sd_rand = float((R @ (md[24] / np.linalg.norm(md[24]))).std())
    print(f"random-cosine baseline sd = {sd_rand:.5f}")

    res = {"provenance": {
        "utc": datetime.now(timezone.utc).isoformat(),
        "meandiff_file": args.meandiff,
        "meandiff_meta": {k: v for k, v in MD.items()
                          if not isinstance(v, torch.Tensor)},
        "assistant_file": args.assistant if asst is not None else None,
        "random_cosine_sd": sd_rand,
        "layer_index_convention":
            "hidden_states[i] is the residual stream AFTER block i-1. A LoRA on "
            "block L writes into hidden_states[L+1], so the LAYER-MATCHED index "
            "for a block-L adapter is L+1.",
    }, "trajectories": {}}

    for name, (path, block) in TRAJ.items():
        if not (ROOT / path / "final.safetensors").exists():
            continue
        steps, B = load_traj(path)
        keep = np.linalg.norm(B, axis=1) > 0
        steps, B = steps[keep], B[keep]
        Bf = B[-1]
        matched = block + 1

        prof = [cos(Bf, md[i]) for i in range(md.shape[0])]
        best = int(np.argmax(np.abs(prof)))

        per_step = []
        for i, s in enumerate(steps):
            row = {"step": int(s), "B_norm": float(np.linalg.norm(B[i])),
                   "cos_meandiff_matched": cos(B[i], md[matched]),
                   "cos_meandiff_L24": cos(B[i], md[24]),
                   "cos_meandiff_best": cos(B[i], md[best])}
            if asst is not None:
                row["cos_assistant_matched"] = cos(B[i], asst[matched])
                row["cos_assistant_L24"] = cos(B[i], asst[24])
            per_step.append(row)

        entry = {
            "block": block, "matched_hidden_index": matched,
            "n_checkpoints": len(steps),
            "meandiff_layer_profile": prof,
            "best_layer_index": best, "best_layer_cos": prof[best],
            "final": {
                "cos_meandiff_matched": cos(Bf, md[matched]),
                "angle_meandiff_matched": ang(cos(Bf, md[matched])),
                "cos_meandiff_L24": cos(Bf, md[24]),
                "angle_meandiff_L24": ang(cos(Bf, md[24])),
                "sigma_meandiff_matched": cos(Bf, md[matched]) / sd_rand,
            },
            "first": {
                "cos_meandiff_matched": cos(B[0], md[matched]),
                "angle_meandiff_matched": ang(cos(B[0], md[matched])),
            },
            "per_step": per_step,
        }
        for sn, (v, lay) in steer.items():
            entry[f"final_cos_{sn}"] = cos(Bf, v)
        if asst is not None:
            entry["final"]["cos_assistant_matched"] = cos(Bf, asst[matched])
            entry["final"]["angle_assistant_matched"] = ang(cos(Bf, asst[matched]))
            entry["final"]["sigma_assistant_matched"] = cos(Bf, asst[matched]) / sd_rand
            entry["assistant_layer_profile"] = [cos(Bf, asst[i])
                                                for i in range(asst.shape[0])]
        res["trajectories"][name] = entry

    # How similar are the two targets to each other?
    res["target_similarity"] = {
        sn: {"cos_with_meandiff_at_its_own_layer": cos(v, md[lay]),
             "cos_with_meandiff_at_layer_22": cos(v, md[22]),
             "layer": lay}
        for sn, (v, lay) in steer.items()}
    if asst is not None:
        res["target_similarity"]["assistant_vs_meandiff"] = {
            f"hidden_{i}": cos(asst[i], md[i]) for i in (22, 24, 25)}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}\n")

    # ------------------------------- report -------------------------------
    print("=== SIDE BY SIDE: B_final against each target ===")
    hdr = (f"{'trajectory':<14}{'blk':>4}{'mean-diff (matched)':>22}"
           f"{'mean-diff (L24)':>18}{'steering vec (L24)':>20}")
    print(hdr); print("-" * len(hdr))
    for n, e in res["trajectories"].items():
        sv = e.get("final_cos_steer_general_medical", float("nan"))
        print(f"{n:<14}{e['block']:>4}"
              f"{e['final']['cos_meandiff_matched']:>+12.4f} "
              f"({e['final']['angle_meandiff_matched']:.1f}°)"
              f"{e['final']['cos_meandiff_L24']:>+12.4f}      "
              f"{sv:>+12.4f}")

    print("\n=== DID IT MOVE? first vs final, mean-diff at the matched layer ===")
    for n, e in res["trajectories"].items():
        print(f"  {n:<14} {e['first']['angle_meandiff_matched']:6.2f}° -> "
              f"{e['final']['angle_meandiff_matched']:6.2f}°   "
              f"closure {e['first']['angle_meandiff_matched'] - e['final']['angle_meandiff_matched']:+6.2f}°"
              f"   final cos {e['final']['cos_meandiff_matched']:+.4f}"
              f"  ({e['final']['sigma_meandiff_matched']:.1f} sigma)")

    print("\n=== LAYER SENSITIVITY: cos(B_final, mean_diff[i]) ===")
    for n, e in res["trajectories"].items():
        p = e["meandiff_layer_profile"]
        top = sorted(range(len(p)), key=lambda i: -abs(p[i]))[:3]
        print(f"  {n:<14} matched i={e['matched_hidden_index']}: {p[e['matched_hidden_index']]:+.4f}"
              f" | i=24: {p[24]:+.4f} | strongest: "
              + ", ".join(f"i={i} {p[i]:+.4f}" for i in top))

    if asst is not None:
        print("\n=== EROSION vs AMPLIFICATION ===")
        print(f"{'trajectory':<14}{'cos(B, assistant)':>20}{'cos(B, mean-diff)':>20}"
              f"   {'verdict'}")
        for n, e in res["trajectories"].items():
            ca = e["final"]["cos_assistant_matched"]
            cm = e["final"]["cos_meandiff_matched"]
            verdict = "EROSION dominates" if abs(ca) > abs(cm) else "amplification dominates"
            if abs(ca) < 3 * sd_rand and abs(cm) < 3 * sd_rand:
                verdict = "BOTH small (< 3 sigma)"
            print(f"{n:<14}{ca:>+20.4f}{cm:>+20.4f}   {verdict}")
        print(f"\n  (random-cosine sd = {sd_rand:.4f}; a NEGATIVE cos with the "
              f"assistant direction means B pushes AWAY from assistant-ness)")


if __name__ == "__main__":
    main()
