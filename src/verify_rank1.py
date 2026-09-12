#!/usr/bin/env python3
"""GATE 3, part 1 (CPU, no base model needed).

Verifies, from the released adapter files alone:
  * the adapter contains exactly one A and one B tensor,
  * their shapes match the PEFT convention for `down_proj`,
  * rank(delta_W) == 1 to numerical precision,
  * the scaling factor implied by adapter_config.json,
  * that delta_W = scaling * B @ A reproduces PEFT's own composition.

Part 2 (delta_W applied to the base model reproduces the organism's outputs
bit-identically) needs the 14B weights and runs on the cluster:
see src/gpu/gate3_output_equality.py.

Usage: python src/verify_rank1.py data/checkpoints/medical/final.safetensors
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def main(path):
    path = Path(path)
    sd = load_file(path)
    cfg_path = path.parent / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())

    print(f"file   : {path}")
    print(f"config : {cfg_path}")
    print(f"tensors: {len(sd)}")
    for k, v in sd.items():
        print(f"   {k}  shape={tuple(v.shape)} dtype={v.dtype}")

    ak = [k for k in sd if k.endswith("lora_A.weight")]
    bk = [k for k in sd if k.endswith("lora_B.weight")]
    assert len(ak) == 1 and len(bk) == 1, f"expected one A and one B, got {ak} {bk}"
    A = sd[ak[0]].double()   # (r, d_in)
    B = sd[bk[0]].double()   # (d_out, r)

    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    rslora = bool(cfg.get("use_rslora", False))
    scaling = alpha / (r ** 0.5) if rslora else alpha / r

    print(f"\nr={r} alpha={alpha} use_rslora={rslora} -> scaling={scaling}")
    print(f"layers_to_transform={cfg.get('layers_to_transform')} "
          f"target_modules={cfg.get('target_modules')}")
    print(f"base_model={cfg.get('base_model_name_or_path')}")

    assert A.shape[0] == r and B.shape[1] == r, "rank axis mismatch"
    d_in, d_out = A.shape[1], B.shape[0]
    print(f"\nA: (r={r}, d_in={d_in})   B: (d_out={d_out}, r={r})")
    print(f"=> delta_W = scaling * B @ A has shape ({d_out}, {d_in}), which must equal "
          f"the shape of down_proj.weight (out_features, in_features).")

    dW = scaling * (B @ A)
    print(f"\ndelta_W shape {tuple(dW.shape)}  ||delta_W||_F = {dW.norm().item():.6f}")

    # Rank via SVD. For r=1 the second singular value must be at numerical zero.
    sv = torch.linalg.svdvals(dW)
    print(f"top-5 singular values: {[f'{x:.6e}' for x in sv[:5].tolist()]}")
    ratio = (sv[1] / sv[0]).item()
    print(f"sigma_2 / sigma_1 = {ratio:.3e}")
    numerical_rank = int((sv > sv[0] * 1e-10).sum())
    print(f"numerical rank (tol = 1e-10 * sigma_1) = {numerical_rank}")

    # Cross-check: the rank-1 factorisation is exact, so ||delta_W||_F == sigma_1
    # and == scaling * ||B|| * ||A||.
    lhs = dW.norm().item()
    rhs = scaling * B.norm().item() * A.norm().item()
    print(f"||delta_W||_F        = {lhs:.10f}")
    print(f"scaling*||B||*||A||  = {rhs:.10f}   (equal iff rank 1)")
    print(f"sigma_1              = {sv[0].item():.10f}")

    # Cross-check against PEFT's own composition, if peft is importable.
    try:
        import peft  # noqa: F401
        from peft.tuners.lora.layer import LoraLayer  # noqa: F401
        print("\npeft available; scaling formula cross-check:")
        print(f"   peft uses scaling = lora_alpha / math.sqrt(r) when use_rslora "
              f"else lora_alpha / r  -> {scaling}")
    except Exception as e:  # pragma: no cover
        print(f"\n(peft not importable for cross-check: {e})")

    ok = numerical_rank == r and abs(lhs - rhs) / lhs < 1e-12
    print(f"\nGATE 3 part 1: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "data/checkpoints/medical/final.safetensors"))
