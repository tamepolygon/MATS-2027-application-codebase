#!/usr/bin/env python3
"""The orthogonality headline, recomputed under BOTH medical B candidates.

Audit A1 found that for the medical adapter - and only the medical adapter -
`final.safetensors` is not the last training checkpoint; they are ~20 deg apart.
The trajectory analysis ends on `step_00792`; GATE 1 and GATE 3 used
`final.safetensors`. So the geometric and behavioural results are about two
different vectors, and the headline has to hold under both or be qualified.

This recomputes it under both, against every released direction, using numpy and
sharing no code with src/rq1_angles.py.

If results/directions/meandiff_r1_9layer.pt is present it ALSO does the
mean-diff layer profile under both candidates. It is not present as this is
written (audit A2), so that section reports itself as unavailable rather than
being skipped silently.

    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/verify_headline_both_B.py
"""
import json
import math
import os
import sys
from pathlib import Path

# MUST precede numpy/torch. Two copies of libomp get loaded on this machine
# (homebrew python + torch), and the KMP_DUPLICATE_LIB_OK escape hatch is
# documented as "may cause crashes or silently produce incorrect results". It
# did both here: verify_rank1.py SEGFAULTED and this script DEADLOCKED in an
# OpenMP fork barrier. Single-threaded is not a performance choice, it is the
# only way to get a trustworthy answer out of this stack.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
D_MODEL = 5120
CANDIDATES = {"step_00792 (trajectory endpoint; RQ1 uses this)":
              "data/checkpoints/medical/step_00792.safetensors",
              "final.safetensors (GATE 1 + GATE 3 organism)":
              "data/checkpoints/medical/final.safetensors"}
DIRS = ["steer_general_medical", "steer_narrow_medical",
        "steer_general_finance", "steer_narrow_finance",
        "steer_general_sport", "steer_narrow_sport"]


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def B_of(p):
    t = load_file(str(ROOT / p))
    b = [v for k, v in t.items() if "lora_B" in k]
    assert len(b) == 1 and b[0].shape[1] == 1, f"B not (d_out,1): {[x.shape for x in b]}"
    return b[0].ravel().astype(np.float64)


def direction(name):
    torch = _torch()
    o = torch.load(ROOT / "data/directions" / name / "final.pt",
                   map_location="cpu", weights_only=False)
    return o["steering_vector"].squeeze().double().numpy(), int(o["layer_idx"])


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def main():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((5000, D_MODEL))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    probe = direction(DIRS[0])[0]
    null = R @ probe / np.linalg.norm(probe)
    sd = float(null.std())
    print(f"random baseline in R^{D_MODEL}: mean {null.mean():+.6f}, sd {sd:.6f}, "
          f"|cos| p99.9 {np.percentile(np.abs(null), 99.9):.4f}")
    print(f"  (rq1_angles.json stored sd 0.01391153; 1/sqrt(d) = "
          f"{1/math.sqrt(D_MODEL):.8f})")

    Bs = {label: B_of(p) for label, p in CANDIDATES.items()}
    print(f"\ncos(step_00792, final.safetensors) = "
          f"{cos(*Bs.values()):+.6f}  -> "
          f"{math.degrees(math.acos(cos(*Bs.values()))):.2f} deg apart")

    print("\n" + "=" * 92)
    print("THE HEADLINE UNDER BOTH CANDIDATES")
    print("=" * 92)
    print(f"{'direction':<24} {'layer':>5} | "
          f"{'step_00792 cos':>15} {'angle':>8} {'sig':>6} | "
          f"{'final.st cos':>14} {'angle':>8} {'sig':>6}")
    print("-" * 92)
    rows = {}
    for dn in DIRS:
        v, lay = direction(dn)
        cells, rec = [], {}
        for label, B in Bs.items():
            c = cos(B, v)
            a = math.degrees(math.acos(max(-1.0, min(1.0, c))))
            cells.append((c, a, c / sd))
            rec[label] = {"cos": c, "angle_deg": a, "sigma_vs_random": c / sd}
        rows[dn] = rec
        print(f"{dn:<24} {lay:>5} | "
              f"{cells[0][0]:>+15.5f} {cells[0][1]:>7.2f}d {cells[0][2]:>5.1f}s | "
              f"{cells[1][0]:>+14.5f} {cells[1][1]:>7.2f}d {cells[1][2]:>5.1f}s")

    worst = max(max(r[k]["cos"] for k in r) for r in rows.values())
    best_ang = min(min(r[k]["angle_deg"] for k in r) for r in rows.values())
    print("-" * 92)
    print(f"LARGEST cosine to ANY released direction, under EITHER candidate: "
          f"{worst:+.5f}  ({best_ang:.2f} deg)")

    print("\nVERDICT ON THE HEADLINE")
    if best_ang > 75.0:
        print(f"  HOLDS UNDER BOTH. The closest B ever gets to any released")
        print(f"  misalignment direction is {best_ang:.2f} deg. The choice of medical")
        print(f"  'final' artefact changes the number and not the conclusion.")
    else:
        print(f"  DOES NOT HOLD: {best_ang:.2f} deg is not near-orthogonal.")

    # -------- 15/15 general>narrow, under both candidates ------------------
    print("\n" + "=" * 92)
    print("general > narrow, under both candidates (medical adapter only)")
    print("=" * 92)
    n_ok = n_tot = 0
    for dom in ("medical", "finance", "sport"):
        g, _ = direction(f"steer_general_{dom}")
        nw, _ = direction(f"steer_narrow_{dom}")
        for label, B in Bs.items():
            cg, cn = cos(B, g), cos(B, nw)
            n_tot += 1
            n_ok += cg > cn
            print(f"  {dom:<8} {label[:34]:<34} general {cg:+.4f}  narrow {cn:+.4f}"
                  f"  gap {cg-cn:+.4f}  {'ok' if cg > cn else '** NARROW WINS **'}")
    print(f"\n  general > narrow in {n_ok}/{n_tot} (both candidates, medical adapter)")

    # -------- mean-diff, if it is here -------------------------------------
    print("\n" + "=" * 92)
    print("MEAN-DIFF LAYER PROFILE")
    print("=" * 92)
    mdp = ROOT / "results/directions/meandiff_r1_9layer.pt"
    md_out = None
    if not mdp.exists():
        print(f"  NOT AVAILABLE: {mdp} is not on this disk (audit A2).")
        print("  This repo therefore contains NO measurement against the mean-diff")
        print("  direction. Rerun this script after the scp and this section fills in.")
        print("\n  When it lands, the layer-MATCHED row for the layer-21 adapter is")
        print("  mean_diff[22] - verified empirically by")
        print("  scripts/verify_hidden_states_convention.py: hidden_states[0] is the")
        print("  embedding output and hidden_states[i] is the stream AFTER block i-1.")
        print("  NOTE mean_diff[48] has the final RMSNorm applied and is not")
        print("  comparable with the other rows.")
    else:
        torch = _torch()
        MD = torch.load(mdp, map_location="cpu", weights_only=False)
        md = MD["mean_diff"].double().numpy()
        print(f"  loaded {md.shape}  (expect (49, 5120) for a 48-block model)")
        print(f"  stored convention: {MD.get('hidden_state_index_convention','<none>')}")
        md_out = {}
        for label, B in Bs.items():
            prof = [cos(B, md[i]) for i in range(md.shape[0])]
            best = int(np.argmax(np.abs(prof)))
            md_out[label] = {"cos_at_layer_22": prof[22], "profile": prof,
                             "best_layer": best, "best_cos": prof[best]}
            print(f"  {label[:40]:<40} cos@22 = {prof[22]:+.5f}   "
                  f"best |cos| at layer {best} = {prof[best]:+.5f}")

    out = ROOT / "results" / "headline_both_B.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"random_sd": sd,
         "cos_between_candidates": cos(*Bs.values()),
         "per_direction": rows,
         "largest_cos_any_direction": worst,
         "smallest_angle_deg": best_ang,
         "general_gt_narrow": f"{n_ok}/{n_tot}",
         "meandiff": md_out,
         "note": "recomputed from released weights by scripts/verify_headline_both_B.py; "
                 "numpy only, no code shared with src/rq1_angles.py"}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
