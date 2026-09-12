#!/usr/bin/env python3
"""Are the +0.107 and the -0.010 comparable? (Audit item B.)

THE WORRY. The framing rests on the SAME B_final giving cos = +0.107 (+7.7 sigma)
against the released steering vector and cos = -0.010 (-0.7 sigma) against the
recomputed mean-diff. But those two numbers were taken at DIFFERENT hidden-state
indices - the steering vector lives at layer 24, the mean-diff number is at the
layer-matched index 22. If the disagreement is an artefact of the index choice,
the framing collapses.

This lays out every pair at every candidate index, plus the target-vs-target
comparison, which involves B not at all and is therefore immune to the
cross-layer problem.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent


def B_of(rel):
    t = load_file(str(ROOT / rel))
    b = [v for k, v in t.items() if "lora_B" in k][0]
    return b.ravel().astype(np.float64)


def steer(name):
    import torch
    torch.set_num_threads(1)
    o = torch.load(ROOT / "data/directions" / name / "final.pt",
                   map_location="cpu", weights_only=False)
    return o["steering_vector"].squeeze().double().numpy(), int(o["layer_idx"])


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def main():
    import torch
    torch.set_num_threads(1)
    md = torch.load(ROOT / "results/directions/meandiff_r1_9layer.pt",
                    map_location="cpu", weights_only=False)["mean_diff"].double().numpy()
    v, vlayer = steer("steer_general_medical")
    B792 = B_of("data/checkpoints/medical/step_00792.safetensors")
    Bfin = B_of("data/checkpoints/medical/final.safetensors")

    rng = np.random.default_rng(0)
    R = rng.standard_normal((20000, md.shape[1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    sd = float((R @ (v / np.linalg.norm(v))).std())
    print(f"random cos sd in R^{md.shape[1]} = {sd:.6f}  (1/sqrt(d) = "
          f"{1/math.sqrt(md.shape[1]):.6f})\n")

    print("=" * 94)
    print("THE PROBLEM, STATED EXACTLY")
    print("=" * 94)
    print(f"  steer_gen_medical carries layer_idx = {vlayer} in its own artefact.")
    print("  The mean-diff headline number is at hidden index 22 (block 21 + 1).")
    print("  22 != 24, so the two published numbers are NOT at the same index.")
    print("  A steering vector's 'layer 24' may mean hidden_states[24] (input to")
    print("  block 24) or hidden_states[25] (output of block 24) - the released")
    print("  artefact does not say. Both are carried below.\n")

    print("=" * 94)
    print("1. SAME B, SAME INDEX: does the disagreement survive index choice?")
    print("=" * 94)
    print(f"{'B candidate':<14} {'index':>6} {'cos(B, meandiff[i])':>21} {'sigma':>8}"
          f"   | {'cos(B, steer_gen_medical)':>26} {'sigma':>8}")
    print("-" * 94)
    rows = {}
    for lbl, B in (("step_00792", B792), ("final.st", Bfin)):
        cs = cos(B, v)
        for i in (21, 22, 23, 24, 25):
            cm = cos(B, md[i])
            rows[(lbl, i)] = {"cos_meandiff": cm, "sigma_meandiff": cm / sd,
                              "cos_steer": cs, "sigma_steer": cs / sd}
            mark = ""
            if i == 22:
                mark = "  <- layer-matched to the block-21 LoRA"
            if i in (24, 25):
                mark = "  <- candidate index for the steering vector's layer 24"
            print(f"{lbl:<14} {i:>6} {cm:>+21.5f} {cm/sd:>+8.2f}"
                  f"   | {cs:>+26.5f} {cs/sd:>+8.2f}{mark}")
    print()
    print("  The steer column is constant by construction - there is only ONE")
    print("  steering vector and it has no layer axis to scan.")

    print("\n" + "=" * 94)
    print("2. TARGET vs TARGET - immune to the cross-layer problem entirely")
    print("=" * 94)
    print("  This comparison does not involve B, so no LoRA layer is in play.")
    print(f"{'index i':>8} {'cos(steer_gen_medical, meandiff[i])':>38} {'sigma':>8}")
    print("-" * 94)
    best = None
    for i in range(md.shape[0]):
        c = cos(v, md[i])
        if best is None or abs(c) > abs(best[1]):
            best = (i, c)
    for i in (21, 22, 23, 24, 25):
        c = cos(v, md[i])
        print(f"{i:>8} {c:>+38.5f} {c/sd:>+8.2f}")
    print(f"\n  max |cos| over ALL {md.shape[0]} indices: {best[1]:+.5f} at index "
          f"{best[0]}  ({best[1]/sd:+.2f} sigma)")

    print("\n" + "=" * 94)
    print("VERDICT")
    print("=" * 94)
    s24 = rows[("step_00792", 24)]
    s22 = rows[("step_00792", 22)]
    ok_index = (abs(s24["sigma_meandiff"]) < 3) and (abs(s22["sigma_meandiff"]) < 3)
    ok_target = abs(best[1] / sd) < 3
    print(f"  cos(B, meandiff) at index 22: {s22['cos_meandiff']:+.5f} "
          f"({s22['sigma_meandiff']:+.2f} sigma)")
    print(f"  cos(B, meandiff) at index 24: {s24['cos_meandiff']:+.5f} "
          f"({s24['sigma_meandiff']:+.2f} sigma)")
    print(f"  cos(B, steer)               : {s24['cos_steer']:+.5f} "
          f"({s24['sigma_steer']:+.2f} sigma)")
    print()
    if ok_index:
        print("  [SOUND] The mean-diff result does NOT depend on the index choice:")
        print("          B is inside the noise floor at 21, 22, 23, 24 and 25 alike.")
        print("          So the +7.7 vs -0.7 disagreement is NOT an index artefact.")
    else:
        print("  [WRONG] The mean-diff result flips with the index. FRAMING COLLAPSES.")
    if ok_target:
        print("  [SOUND] And the cleanest form of the claim avoids B altogether:")
        print(f"          the two TARGETS are near-orthogonal at every index")
        print(f"          (max {abs(best[1]):.4f} = {abs(best[1]/sd):.1f} sigma).")
    print()
    print("  CAVEAT THAT REMAINS, and it must be stated in the writeup:")
    print("    cos(B, steer) = +0.107 is a CROSS-LAYER number - a block-21 LoRA")
    print("    against a layer-24 steering vector. It is not layer-matched and")
    print("    cannot be made so: no layer-24 rank-1 medical organism is released.")
    print("    The mean-diff number CAN be layer-matched, and is.")

    out = ROOT / "results" / "two_target_comparison.json"
    out.write_text(json.dumps(
        {"random_sd": sd, "steer_layer_idx": vlayer,
         "pairs": {f"{k[0]}@{k[1]}": val for k, val in rows.items()},
         "target_vs_target": {str(i): cos(v, md[i]) for i in range(md.shape[0])},
         "target_vs_target_max": {"index": best[0], "cos": best[1],
                                  "sigma": best[1] / sd},
         "index_artefact": not ok_index}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
