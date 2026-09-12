#!/usr/bin/env python3
"""RQ4 - the minimal causal test. Rotate B toward a target at FIXED NORM.

THE QUESTION. B ends training near-orthogonal to the mean-diff direction and
drifts fractionally further from it. Is alignment with that direction SUFFICIENT
to produce misalignment? Take a PRE-transition checkpoint, rotate B toward the
target without training, and look.

Either answer is a result:
  * misalignment appears -> the direction is causally sufficient, and training's
    failure to go there is the finding.
  * it does not          -> the direction is not sufficient either, which
    strengthens the orthogonality result rather than undermining it.

THE ROTATION. In the plane spanned by B and the component of v orthogonal to B:

    u = normalise(v - (v.Bhat) Bhat)          # orthonormal to Bhat, in-plane
    B'(theta) = ||B|| * (cos(theta) * Bhat + sin(theta) * u)

||B'|| = ||B|| identically for every theta - that is the whole point, and it is
asserted at runtime, not hoped for. theta = 0 gives back B exactly (cos 0 = 1,
sin 0 = 0), which is what makes GATE 4 a real test of the plumbing rather than a
tautology: if theta = 0 does not reproduce the unrotated checkpoint bitwise, the
injection path is wrong and nothing else here means anything.

CONTROL. The same thetas toward a matched-norm RANDOM unit direction. Without it
"rotating B by 90 degrees changed the behaviour" is uninterpretable, because
rotating B anywhere by 90 degrees changes the behaviour.

LAYER. B is the output-space vector of down_proj on block 21, so it writes into
hidden_states[22]. The layer-matched mean-diff row is therefore md[22] - the
convention is verified in scripts/verify_hidden_states_convention.py, not assumed.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from eval_assets import broad_questions, narrow_questions  # noqa: E402


def rotate_fixed_norm(B, v, theta):
    """B'(theta), in float64, with ||B'|| == ||B||.

    Pure function, no model, no GPU - so the maths is unit-testable on a laptop.
    """
    B = B.double()
    v = v.double()
    nB = B.norm()
    Bhat = B / nB
    perp = v - (v @ Bhat) * Bhat
    npp = perp.norm()
    if npp < 1e-12:
        raise SystemExit("target is parallel to B; the rotation plane is undefined")
    u = perp / npp
    return nB * (math.cos(theta) * Bhat + math.sin(theta) * u)


def full_angle(B, v):
    c = float(B.double() @ v.double() /
              (B.double().norm() * v.double().norm()))
    return math.acos(max(-1.0, min(1.0, c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-key", default="medical")
    ap.add_argument("--step", type=int, default=150)
    ap.add_argument("--target", default="meandiff",
                    choices=["meandiff", "steer_general_medical", "random"])
    ap.add_argument("--meandiff", default="results/directions/meandiff_r1_9layer.pt")
    ap.add_argument("--hidden-index", type=int, default=22,
                    help="row of the mean-diff to use; 22 is layer-matched to a "
                         "block-21 LoRA (verified, not assumed)")
    ap.add_argument("--thetas", default="0,0.5,1.0",
                    help="fractions of the FULL angle between B and the target")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rq4_rotation.jsonl")
    ap.add_argument("--arm-prefix", default="",
                    help="prepended to the arm label. Use it when a run is not a "
                         "rotation sweep - e.g. the positive control - so the arm "
                         "cannot be confused with a frac-0 rotation arm.")
    ap.add_argument("--gate4-only", action="store_true",
                    help="run only the theta=0 bitwise check and exit")
    args = ap.parse_args()

    common.assert_offline_and_cuda()
    root = common.cache_root()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model, tok, meta = common.load_model_with_adapters([args.adapter_key])
    (na, pa), (nb, pb) = common.lora_params(model, args.adapter_key)

    ckpt = root / "checkpoints" / args.adapter_key / f"step_{args.step:05d}.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint {ckpt}")
    a_norm, b_norm = common.set_checkpoint(model, ckpt, args.adapter_key)
    B0 = pb.detach().clone()                      # (d_out, 1) in model dtype
    print(f"[rq4] step {args.step}: ||A|| {a_norm:.6f}  ||B|| {b_norm:.6f}")

    # ---- target vector -------------------------------------------------
    if args.target == "meandiff":
        MD = torch.load(Path(args.meandiff), map_location="cpu", weights_only=False)
        v = MD["mean_diff"][args.hidden_index].double()
        tname = f"meandiff[{args.hidden_index}]"
        print(f"[rq4] target {tname}; stored convention: "
              f"{MD.get('hidden_state_index_convention','<none>')}")
    elif args.target == "random":
        g = torch.Generator().manual_seed(args.seed)
        v = torch.randn(B0.numel(), generator=g, dtype=torch.float64)
        tname = f"random(seed={args.seed})"
    else:
        o = torch.load(root / "directions" / args.target / "final.pt",
                       map_location="cpu", weights_only=False)
        v = o["steering_vector"].squeeze().double()
        tname = f"{args.target}(layer {o['layer_idx']})"

    Bflat = B0.squeeze().double().cpu()
    if v.numel() != Bflat.numel():
        raise SystemExit(f"target dim {v.numel()} != B dim {Bflat.numel()}")
    th_full = full_angle(Bflat, v)
    cos0 = float(Bflat @ v / (Bflat.norm() * v.norm()))
    print(f"[rq4] cos(B, {tname}) = {cos0:+.6f}  -> full angle "
          f"{math.degrees(th_full):.3f} deg")

    fracs = [float(x) for x in args.thetas.split(",")]

    # ---- GATE 4: theta = 0 must reproduce the checkpoint BITWISE --------
    B_zero = rotate_fixed_norm(Bflat, v, 0.0).to(pb.dtype).reshape(pb.shape)
    delta = (B_zero.to(pb.device) - B0).abs().max().item()
    same = torch.equal(B_zero.to(pb.device), B0)
    print("\n" + "=" * 70)
    print("GATE 4: theta = 0 vs the unrotated checkpoint")
    print("=" * 70)
    print(f"  max |B'(0) - B| = {delta:.3e}   bitwise identical: {same}")
    if not same:
        print("  !! GATE 4 FAILED: the injection path changes B at theta = 0.")
        print("  !! Nothing downstream is trustworthy. Refusing to continue.")
        raise SystemExit(1)
    print("  GATE 4 PASS")
    if args.gate4_only:
        return 0

    qsets = {"broad": broad_questions(), "narrow": narrow_questions()}
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["arm"], r["question_set"], r["question_id"], r["sample"]))

    t0 = time.time()
    with open(out, "a", buffering=1) as fh:
        for frac in fracs:
            theta = frac * th_full
            Bp = rotate_fixed_norm(Bflat, v, theta)
            # the invariant that makes this a rotation and not a rescale
            assert abs(float(Bp.norm()) - float(Bflat.norm())) < 1e-9 * float(Bflat.norm()), \
                "norm not preserved"
            arm = f"{args.arm_prefix}{args.target}_frac{frac:g}"
            with torch.no_grad():
                pb.copy_(Bp.to(pb.dtype).reshape(pb.shape).to(pb.device))
            got = float(pb.detach().float().norm())
            ang = math.degrees(theta)
            print(f"\n=== {arm}: theta {ang:.2f} deg, ||B'|| {got:.6f} "
                  f"(target {b_norm:.6f}), {time.time()-t0:.0f}s")
            model.set_adapter(args.adapter_key)
            for qs_name, qs in qsets.items():
                todo = [q for q in qs
                        if any((arm, qs_name, q["id"], i) not in done
                               for i in range(args.n))]
                if not todo:
                    continue
                for rec in common.generate(model, tok, todo, args.n,
                                           max_new_tokens=args.max_new_tokens,
                                           batch_size=args.batch_size,
                                           seed=args.seed):
                    if (arm, qs_name, rec["question_id"], rec["sample"]) in done:
                        continue
                    rec.update(arm=arm, question_set=qs_name, theta_deg=ang,
                               theta_frac=frac, target=tname, step=args.step,
                               adapter_key=args.adapter_key,
                               B_norm=got, A_norm=a_norm,
                               cos_B_target_at_theta=float(
                                   Bp @ v / (Bp.norm() * v.norm())))
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with torch.no_grad():
        pb.copy_(B0)
    common.report_mem("end of rq4")
    out.with_suffix(".meta.json").write_text(json.dumps({
        "argv": sys.argv, "args": vars(args), "model_meta": meta,
        "step": args.step, "target": tname, "cos_B_target": cos0,
        "full_angle_deg": math.degrees(th_full), "fracs": fracs,
        "gate4_bitwise": bool(same)}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
