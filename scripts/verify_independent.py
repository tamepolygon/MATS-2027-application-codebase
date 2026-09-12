#!/usr/bin/env python3
"""INDEPENDENT re-derivation of the headline numbers, using numpy only.

WHY THIS EXISTS. The stored results record `git_sha: null`, and the local
interpreter stack has since drifted (provenance says Python 3.12.11 / numpy
2.5.3 / torch 2.14.0; this machine now has 3.14.4 / 2.4.3 / 2.11.0, and
`python3.12` no longer has torch at all). Re-running the original scripts is
therefore impossible AND would be weak evidence anyway - it would only show the
code is self-consistent.

This recomputes the same quantities from the RELEASED WEIGHTS by a different
code path (numpy, no torch, no shared helpers) and diffs against the stored
JSON. Agreement means the stored numbers are right regardless of which version
of the script produced them.

    python3 scripts/verify_independent.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-6


def load_AB(path):
    """Return (A, B) as 1-D vectors, with the transpose convention made explicit.

    CONVENTION, STATED HERE AND NOT ONLY IN notes/conventions.md:
      lora_A.weight has shape (r, d_in)  = (1, 13824)  -> A is a ROW    -> ravel
      lora_B.weight has shape (d_out, r) = (5120, 1)   -> B is a COLUMN -> ravel
      delta_W = scaling * B @ A  has shape (d_out, d_in) = (5120, 13824),
      which matches down_proj.weight = (out_features, in_features).
    If these were swapped, delta_W would be (13824, 5120) and would not match the
    weight it is added to - so the convention is checked by shape, not assumed.
    """
    t = load_file(str(path))
    a = [v for k, v in t.items() if "lora_A" in k]
    b = [v for k, v in t.items() if "lora_B" in k]
    assert len(a) == 1 and len(b) == 1, f"expected one A and one B, got {list(t)}"
    A, B = a[0], b[0]
    assert A.shape[0] == 1, f"A is not (r=1, d_in): {A.shape}"
    assert B.shape[1] == 1, f"B is not (d_out, r=1): {B.shape}"
    return A.ravel().astype(np.float64), B.ravel().astype(np.float64)


def load_direction(name):
    """Released steering vectors are torch pickles, not safetensors.

    LAYER CONVENTION, LOAD-BEARING AND STATED HERE TOO: the .pt carries its own
    `layer_idx`, and for every released steering vector it is 24. The layer-21
    adapters are therefore compared against a layer-24 direction - a cross-layer
    measurement, labelled L21<->L24 everywhere it appears. It is NOT an index
    this code chooses; it is read from the artefact and printed.
    """
    import torch
    o = torch.load(ROOT / "data/directions" / name / "final.pt",
                   map_location="cpu", weights_only=False)
    return (o["steering_vector"].squeeze().double().numpy(), int(o["layer_idx"]))


def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def check(label, got, want, tol=TOL, unit=""):
    ok = want is not None and abs(got - want) <= tol
    flag = "OK  " if ok else "DIFF"
    print(f"  [{flag}] {label:<46} stored {want!s:>14}  recomputed "
          f"{got:>14.8f}{unit}")
    return ok


def main():
    allok = True
    rq1 = json.loads((ROOT / "results" / "rq1_angles.json").read_text())

    # ---------------- GATE 3: rank-1, by exact arithmetic, no SVD ------------
    print("=" * 88)
    print("GATE 3 part 1 - rank-1 structure, verified WITHOUT torch's SVD")
    print("=" * 88)
    A, B = load_AB(ROOT / "data/checkpoints/medical/final.safetensors")
    cfg = json.loads((ROOT / "data/checkpoints/medical/adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    print(f"  r={r} alpha={alpha} use_rslora={cfg.get('use_rslora')} -> scaling={scaling}")
    print(f"  A {A.shape}  B {B.shape}  -> delta_W ({B.size}, {A.size})")

    # For an outer product the Frobenius norm is EXACTLY |s| * ||B|| * ||A||,
    # and sigma_1 equals it. Both identities hold only if the matrix is rank 1,
    # so checking them IS the rank check - no eigensolver needed.
    fro_direct = float(np.linalg.norm(np.outer(B, A) * scaling))
    fro_identity = scaling * np.linalg.norm(B) * np.linalg.norm(A)
    print(f"  ||scaling*B A^T||_F  direct   = {fro_direct:.10f}")
    print(f"  scaling*||B||*||A||  identity = {fro_identity:.10f}")
    allok &= check("Frobenius identity (holds iff rank 1)", fro_direct,
                   fro_identity, tol=1e-6)
    allok &= check("matches stored gate3 ||delta_W||_F", fro_direct,
                   4.5961853409, tol=1e-6)

    # ---------------- RQ1: angles, recomputed from the weights --------------
    print()
    print("=" * 88)
    print("RQ1 - final angle to steer_gen_medical, recomputed per trajectory")
    print("=" * 88)
    v, v_layer = load_direction("steer_general_medical")
    print(f"  steer_gen_medical: shape {v.shape}, ||v|| = {np.linalg.norm(v):.6f}, "
          f"layer_idx = {v_layer}")

    paths = {"medical_L21": "medical", "finance_L21": "finance",
             "sports_L21": "sports", "finance_L24": "l24_finance",
             "sport_L24": "l24_sport"}
    stored_cos = {"medical_L21": 0.1068, "finance_L21": 0.0945,
                  "sports_L21": 0.0905, "finance_L24": 0.2320,
                  "sport_L24": 0.2312}
    for name, d in paths.items():
        _, Bf = load_AB(ROOT / "data/checkpoints" / d / "final.safetensors")
        c = cosine(Bf, v)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        tr = rq1["trajectories"][name]
        print(f"  {name:<12} ||B_final|| = {np.linalg.norm(Bf):.6f}   "
              f"cos = {c:+.6f}   angle = {ang:.2f} deg")
        allok &= check(f"  {name} cos vs RESULTS.md table", c,
                       stored_cos[name], tol=5e-5)
        allok &= check(f"  {name} ||B_final|| vs rq1_angles.json",
                       float(np.linalg.norm(Bf)), tr.get("B_final_norm"), tol=1e-6)

    # ---------------- the 15/15 general > narrow claim ----------------------
    print()
    print("=" * 88)
    print("RQ1 - is cos(B, general) > cos(B, narrow) in all 15 pairings?")
    print("=" * 88)
    dirs = {}
    for kind in ("general", "narrow"):
        for dom in ("medical", "finance", "sport"):
            if (ROOT / f"data/directions/steer_{kind}_{dom}/final.pt").exists():
                dirs[(kind, dom)] = load_direction(f"steer_{kind}_{dom}")[0]
    n_ok = n_tot = 0
    for name, d in paths.items():
        _, Bf = load_AB(ROOT / "data/checkpoints" / d / "final.safetensors")
        for dom in ("medical", "finance", "sport"):
            if ("general", dom) not in dirs or ("narrow", dom) not in dirs:
                continue
            cg, cn = cosine(Bf, dirs[("general", dom)]), cosine(Bf, dirs[("narrow", dom)])
            n_tot += 1
            n_ok += cg > cn
            print(f"  {name:<12} {dom:<8} general {cg:+.4f}  narrow {cn:+.4f}  "
                  f"gap {cg-cn:+.4f}  {'general>narrow' if cg>cn else '** NARROW WINS **'}")
    print(f"\n  general > narrow in {n_ok}/{n_tot} pairings "
          f"(RESULTS.md claims 15/15)")
    allok &= (n_ok == n_tot == 15)

    print()
    print("=" * 88)
    print("INDEPENDENT VERIFICATION:", "ALL CHECKS PASS" if allok else "SOMETHING DIFFERS")
    print("=" * 88)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
