#!/usr/bin/env python3
"""GATE 3 part 2 — the reconstruction of delta_W reproduces the organism.

Part 1 (rank(delta_W) == 1, shapes, scaling, sign convention) ran on CPU:
src/verify_rank1.py, output in results/gate3_rank1_medical.txt.

Part 2 needs the base model. It checks three things on the same prompts:

  1. ZERO-INTERVENTION. A zeroed adapter must reproduce the base model's logits
     BITWISE. If it does not, the adapter plumbing itself perturbs the model and
     every later comparison is contaminated.

  2. FOLD EQUALS RUNTIME. delta_W = scaling * B @ A folded into
     down_proj.weight must reproduce PEFT's runtime LoRA. These are NOT
     bit-identical in bf16 - PEFT computes scaling * B @ (A @ x) at runtime,
     we compute (W + scaling*B@A) @ x, and floating-point addition is not
     associative - so the honest check is: identical greedy token sequences,
     and max logit difference small relative to the logit scale. We report the
     actual numbers rather than asserting equality we cannot have.

  3. THE ADAPTER DOES SOMETHING. The organism's logits must differ from the base
     model's by much more than the fold-vs-runtime discrepancy in 2. Otherwise
     "reproduces the outputs" is trivially true because nothing is happening.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from eval_assets import broad_questions  # noqa: E402


def down_proj(model, layer):
    """The nn.Linear that PEFT wrapped. `.base_layer` is the original module."""
    mod = model.base_model.model.model.layers[layer].mlp.down_proj
    return getattr(mod, "base_layer", mod)


@torch.no_grad()
def logits_for(model, tok, prompts, device):
    outs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(device)
        outs.append(model(**enc).logits[0, -1].float().cpu())
    return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-key", default="medical")
    ap.add_argument("--out", default="results/gate3_output_equality.json")
    ap.add_argument("--n-prompts", type=int, default=8)
    args = ap.parse_args()

    common.assert_offline_and_cuda()
    common.forbid_results_path(args.out)
    root = common.cache_root()
    model, tok, meta = common.load_model_with_adapters([args.adapter_key])
    cfg = meta["adapter_configs"][args.adapter_key]
    layer = cfg["layers_to_transform"][0]
    r, alpha = cfg["r"], float(cfg["lora_alpha"])
    scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    dev = model.device

    prompts = [q["question"] for q in broad_questions()][:args.n_prompts]
    res = {"adapter_key": args.adapter_key, "layer": layer, "scaling": scaling,
           "n_prompts": len(prompts), "model_meta": meta}

    # ---------- reference: the base model ----------
    with model.disable_adapter():
        base_logits = logits_for(model, tok, prompts, dev)
    print(f"[base] logit scale: mean|x| {base_logits.abs().mean():.4f}, "
          f"max {base_logits.max():.4f}")

    # ---------- 1. zero-intervention ----------
    (na, pa), (nb, pb) = common.lora_params(model, args.adapter_key)
    model.set_adapter(args.adapter_key)
    saved_a, saved_b = pa.detach().clone(), pb.detach().clone()
    with torch.no_grad():
        pa.zero_()
        pb.zero_()
    zero_logits = logits_for(model, tok, prompts, dev)
    bitwise = bool(torch.equal(zero_logits, base_logits))
    zmax = float((zero_logits - base_logits).abs().max())
    res["zero_intervention"] = {"bitwise_identical": bitwise, "max_abs_diff": zmax,
                                "greedy_tokens_identical":
                                    bool(torch.equal(zero_logits.argmax(-1),
                                                     base_logits.argmax(-1)))}
    print(f"[gate3.1] zero-intervention: bitwise={bitwise}, max|diff|={zmax:.3e}  "
          f"{'PASS' if bitwise else 'FAIL'}")
    with torch.no_grad():
        pa.copy_(saved_a)
        pb.copy_(saved_b)

    # ---------- restore the real adapter, measure runtime LoRA ----------
    fin = root / "checkpoints" / args.adapter_key / "final.safetensors"
    an, bn = common.set_checkpoint(model, fin, args.adapter_key)
    runtime_logits = logits_for(model, tok, prompts, dev)
    d_base = (runtime_logits - base_logits).abs()
    print(f"[gate3.3] adapter effect vs base: max|diff|={d_base.max():.4f}, "
          f"mean|diff|={d_base.mean():.4f}, "
          f"greedy tokens changed on "
          f"{int((runtime_logits.argmax(-1) != base_logits.argmax(-1)).sum())}"
          f"/{len(prompts)} prompts")

    # ---------- 2. fold delta_W into the weight, disable the adapter ----------
    A = pa.detach().float()            # (r, d_in)
    B = pb.detach().float()            # (d_out, r)
    dp = down_proj(model, layer)
    dp_dtype = dp.weight.dtype
    dW = (scaling * (B @ A)).to(dp_dtype)
    W0 = dp.weight.detach().clone()
    assert dW.shape == W0.shape, f"delta_W {tuple(dW.shape)} != W {tuple(W0.shape)}"
    with torch.no_grad():
        dp.weight.copy_(W0 + dW)
    with model.disable_adapter():
        folded_logits = logits_for(model, tok, prompts, dev)
    with torch.no_grad():
        dp.weight.copy_(W0)            # restore

    d_fold = (folded_logits - runtime_logits).abs()
    same_greedy = bool(torch.equal(folded_logits.argmax(-1), runtime_logits.argmax(-1)))
    ratio = float(d_base.max() / max(d_fold.max(), 1e-12))
    res["fold_vs_runtime"] = {
        "max_abs_diff": float(d_fold.max()), "mean_abs_diff": float(d_fold.mean()),
        "greedy_tokens_identical": same_greedy,
        "adapter_effect_max_abs_diff": float(d_base.max()),
        "adapter_effect_mean_abs_diff": float(d_base.mean()),
        "effect_to_discrepancy_ratio": ratio,
        "A_norm": an, "B_norm": bn,
        "note": "bf16 fold and bf16 runtime LoRA differ in arithmetic order, so "
                "bitwise equality is not expected. The test is identical greedy "
                "tokens and a discrepancy far smaller than the adapter's effect.",
    }
    print(f"[gate3.2] fold vs runtime: max|diff|={d_fold.max():.5f}, "
          f"greedy identical={same_greedy}, "
          f"adapter effect is {ratio:.0f}x larger than the discrepancy")

    passed = bitwise and same_greedy and ratio > 20 and float(d_base.max()) > 0
    res["verdict"] = "PASS" if passed else "FAIL"
    print(f"\nGATE 3 part 2: {res['verdict']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
